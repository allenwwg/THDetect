#pragma once
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include<pybind11/buffer_info.h>
#include <pybind11/embed.h>

#include <string.h>
#include <algorithm>

#define MAX_BOX_NUM 400
namespace py = pybind11;
using py::arg;



struct boxInfo {
	int x1;
	int x2;
	int y1;
	int y2;
	int label_id;
	double score;
	bool isExist = true;
};
bool union_set[MAX_BOX_NUM];

class PatchManager {
public:
	PatchManager(int pscale, int pstep, int ph, int pw):img_scale(pscale),step(pstep),img_h(ph),img_w(pw) {
		margin = (img_scale - step) / 2;
		patch_cols = pw / step;
		patch_rows = ph / step;
		memcpy(findBoxTable, findBoxTable + MAX_BOX_NUM, 0);
	}
public:
	std::string AddPatch(py::array_t<double>& bboxes, py::array_t<int>& label, int row_id, int col_id) {
		if (row_id * patch_cols + col_id - last_index > 1) {
			// 紧挨的上一张为空
			for (int i = last_index + 1; i < row_id * patch_cols + col_id; i++) {
				unsigned short int* tmpcat = new unsigned short int[1];
				unsigned short int* tmpid = new unsigned short int[1];
				unsigned short int* tmpscore = new unsigned short int[1];
				myPool_cat.push_back(tmpcat);
				myPool_id.push_back(tmpid);
				myPool_score.push_back(tmpscore);
				myPool_isValid.push_back(false);
				if (myPool_cat.size() >= patch_cols + 1) {	//queue已满
					if (myPool_isValid.front()) {
						unsigned short int* this_clip_id = myPool_id.front();
						for (int i = 0; i < step * step; i++) {
							boxExistTable[this_clip_id[i]] = true;
						}
					}
					delete[] myPool_cat.front();
					delete[] myPool_id.front();
					delete[] myPool_score.front();
					myPool_cat.pop_front();
					myPool_id.pop_front();
					myPool_score.pop_front();
					myPool_isValid.pop_front();
				}
			}
		}
		last_index = row_id * patch_cols + col_id;

		/* gen_box_mask*/
		unsigned short int* tmpcat = new unsigned short int[step * step]();
		unsigned short int* tmpid = new unsigned short int[step * step]();
		unsigned short int* tmpscore = new unsigned short int[step * step]();

		int origin_x = col_id * step;
		int origin_y = row_id * step;

		py::buffer_info buf1 = bboxes.request();
		py::buffer_info buf2 = label.request();

		if (buf1.ndim != 2 || buf2.ndim != 1)
		{
			throw std::runtime_error("numpy.ndarray dims must be 2!");
		}
		if ((buf1.shape[0] != buf2.shape[0])) {
			throw std::runtime_error("two array shape must be match!");
		}

		double* ptr1 = (double*)buf1.ptr;//指针访问读写 numpy.ndarray
		int* ptr2 = (int*)buf2.ptr;

		for (int i = 0; i < buf1.shape[0]; i++)
		{
			box_cnt += 1;
			struct boxInfo tmpbox;
			unsigned short int t_score = 1;
			tmpbox.x1 = this->clamp(0, img_scale, (int)ptr1[i * buf1.shape[1] + 0]);
			tmpbox.y1 = this->clamp(0, img_scale, (int)ptr1[i * buf1.shape[1] + 1]);
			tmpbox.x2 = this->clamp(0, img_scale, (int)ptr1[i * buf1.shape[1] + 2]);
			tmpbox.y2 = this->clamp(0, img_scale, (int)ptr1[i * buf1.shape[1] + 3]);
			tmpbox.score = ptr1[i * buf1.shape[1] + 4];
			t_score = int(tmpbox.score * 100000 - 40000);
			tmpbox.label_id = ptr2[i];
			int x1, x2, y1, y2;
			x1 = this->clamp(margin, margin + step, tmpbox.x1);
			y1 = this->clamp(margin, margin + step, tmpbox.y1);
			x2 = this->clamp(margin, margin + step, tmpbox.x2);
			y2 = this->clamp(margin, margin + step, tmpbox.y2);

			if (tmpbox.label_id == 11) {
				t_score = 60000;
			}
			else if (tmpbox.label_id == 7) {
				t_score = 60001;
			}

			memcpy(union_set, union_set + MAX_BOX_NUM, 0);
			bool flag = true;
			for (int y = y1 - margin; y < y2 - margin; y++) {
				for (int x = x1 - margin; x < x2 - margin; x++) {
					if (tmpscore[y * step + x] <= t_score) { // FIXME: 如果是同类别的话就不要更新了
						tmpscore[y * step + x] = t_score;
						if (tmpcat[y * step + x] == tmpbox.label_id + 1) {
							union_set[tmpid[y * step + x]] = true;
						}
						tmpcat[y * step + x] = tmpbox.label_id + 1;  // FIXME : 防止0
						tmpid[y * step + x] = box_cnt;
					}
				}
			}

			for (int j = 0; j < step * step; j++) {
				if (union_set[tmpid[j]]) {
					tmpid[j] = box_cnt;
				}
			}

			tmpbox.x1 += origin_x;
			tmpbox.x2 += origin_x;
			tmpbox.y1 += origin_y;
			tmpbox.y2 += origin_y;
			box_dict[box_cnt] = tmpbox;
			box_dict[box_cnt].isExist = true;
		}
		// Push当前块
		myPool_cat.push_back(tmpcat);
		myPool_id.push_back(tmpid);
		myPool_score.push_back(tmpscore);
		myPool_isValid.push_back(true);

		/* MERGE LEFT AND TOP:*/
		merge_clip(row_id, col_id);

		if (myPool_cat.size() >= patch_cols + 1) {	//queue已满
			if (myPool_isValid.front()) {
				unsigned short int* this_clip_id = myPool_id.front();
				for (int i = 0; i < step * step; i++) {
					boxExistTable[this_clip_id[i]] = true;
				}
			}
			delete[] myPool_cat.front();
			delete[] myPool_id.front();
			delete[] myPool_score.front();
			myPool_cat.pop_front();
			myPool_id.pop_front();
			myPool_score.pop_front();
			myPool_isValid.pop_front();
		}
		patch_cnt += 1;
		return std::to_string(patch_cols) + " patch-cols, " + std::to_string(box_cnt) + " boxes";
	}
	int getBoxCnt() {
		return box_cnt;
	}
	void merge_clip(int row_id, int col_id) {
		int index_in_deque = myPool_cat.size() - 1;
		unsigned short int* this_clip_cat = myPool_cat.at(index_in_deque);
		unsigned short int* this_clip_id = myPool_id.at(index_in_deque);
		unsigned short int* this_clip_score = myPool_score.at(index_in_deque);
		unsigned short int* neighbor_clip_cat;
		unsigned short int* neighbor_clip_id;
		unsigned short int* neighbor_clip_score;

		if (col_id > 0 && myPool_isValid.at(index_in_deque - 1)) {
			neighbor_clip_cat = myPool_cat.at(index_in_deque - 1);
			neighbor_clip_id = myPool_id.at(index_in_deque - 1);
			neighbor_clip_score = myPool_score.at(index_in_deque - 1);
			unsigned short int* temp = new unsigned short int[step]();
			int inter_cnt = 0;
			bool cat_set[14];	// 相交的种类
			for (int i = 0; i < step; i++) {
				if (this_clip_cat[i * step] == neighbor_clip_cat[i * step + step - 1] && this_clip_cat[i * step] != 0) {
					temp[i] = this_clip_cat[i * step];	// 相交
					cat_set[temp[i]] = true;
					inter_cnt += 1;
				}
			}
			std::vector<std::pair<int, int> > pair_list;
			for (int item = 0; item < 14; item++) {
				if (!cat_set[item]) continue;
				int i = 0;
				int j = 0;
				while (i < step) {
					j = i;
					while (temp[j] == item) {
						j += 1;
						if (j == step) break;
					}
					if (j != i) {
						pair_list.push_back(std::pair<int, int>(i, j - 1));
					}
					i = j;
					i += 1;
				}
			}
			for (int i = 0; i < pair_list.size(); i++) {
				int first = pair_list[i].first;
				int last = pair_list[i].second;
				int len = last - first + 1;
				int cat = this_clip_cat[first * step];
				while (neighbor_clip_cat[first * step + step - 1] == cat && first >= 0) {
					first -= 1;
				}
				while (neighbor_clip_cat[last * step + step - 1] == cat && last < step) {
					last += 1;
				}
				first += 1;
				last -= 1;
				if (len * 1.0 / (last - first + 1) >= 0.2) {
					int top_id = neighbor_clip_id[(last + first) / 2 * step + step - 1];
					first = pair_list[i].first;
					last = pair_list[i].second;
					int this_id = this_clip_id[(last + first) / 2 * step];
					for (int y = 0; y < step; y++) {
						for (int x = 0; x < step; x++) {
							if (this_clip_id[y * step + x] == this_id && this_clip_cat[y * step + x] == cat) {
								this_clip_id[y * step + x] == top_id;
							}
						}
					}
					if (box_dict[top_id].isExist && box_dict[this_id].isExist) {
						box_dict[top_id] = mergeDict(box_dict[top_id], box_dict[this_id]);
						box_dict[this_id].isExist = false;
						findBoxTable[this_id] = false;
					}
				}

			}
			delete[] temp;
		}

		if (row_id > 0 && myPool_isValid.at(index_in_deque - patch_cols)) {
			neighbor_clip_cat = myPool_cat.at(index_in_deque - patch_cols);
			neighbor_clip_id = myPool_id.at(index_in_deque - patch_cols);
			neighbor_clip_score = myPool_score.at(index_in_deque - patch_cols);
			unsigned short int* temp = new unsigned short int[step]();
			int inter_cnt = 0;
			bool cat_set[14] = { false };	// 相交的种类
			for (int i = 0; i < step; i++) {
				if (this_clip_cat[i] == neighbor_clip_cat[(step - 1) * step + i] && this_clip_cat[i] != 0) {
					temp[i] = this_clip_cat[i];	// 相交
					cat_set[temp[i]] = true;
					inter_cnt += 1;
				}
			}
			std::vector<std::pair<int, int> > pair_list;
			for (int item = 0; item < 14; item++) {
				if (!cat_set[item]) continue;
				int i = 0;
				int j = 0;
				while (i < step) {
					j = i;
					while (temp[j] == item) {
						j += 1;
						if (j == step) break;
					}
					if (j != i) {
						pair_list.push_back(std::pair<int, int>(i, j - 1));
					}
					i = j;
					i += 1;
				}
			}
			for (int i = 0; i < pair_list.size(); i++) {
				int first = pair_list[i].first;
				int last = pair_list[i].second;
				int len = last - first + 1;
				int cat = this_clip_cat[first];
				while (neighbor_clip_cat[(step - 1) * step + first] == cat && first >= 0) {
					first -= 1;
				}
				while (neighbor_clip_cat[(step - 1) * step + last] == cat && last < step) {
					last += 1;
				}
				first += 1;
				last -= 1;
				if (len * 1.0 / (last - first + 1) >= 0.2) {
					int top_id = neighbor_clip_id[(step - 1) * step + (last + first) / 2];
					first = pair_list[i].first;
					last = pair_list[i].second;
					int this_id = this_clip_id[(last + first) / 2];
					for (int y = 0; y < step; y++) {
						for (int x = 0; x < step; x++) {
							if (this_clip_id[y * step + x] == this_id && this_clip_cat[y * step + x] == cat) {
								this_clip_id[y * step + x] == top_id;
							}
						}
					}
					if (box_dict[top_id].isExist && box_dict[this_id].isExist) {
						box_dict[top_id] = mergeDict(box_dict[top_id], box_dict[this_id]);
						box_dict[this_id].isExist = false;
						findBoxTable[this_id] = false;
					}
				}

			}
			delete[] temp;
		}

	}

	struct boxInfo& mergeDict(boxInfo& a, boxInfo& b) {
		boxInfo res;
		res.x1 = std::min(a.x1, b.x1);
		res.y1 = std::min(a.y1, b.y1);
		res.x2 = std::max(a.x2, b.x2);
		res.y2 = std::max(a.y2, b.y2);
		res.score = std::max(a.score, b.score);
		res.label_id = a.label_id;
		return res;
	}

	std::map<int, py::list> getInfo() {
		// 处理还在deque中的部分
		while (!myPool_cat.empty()) {	//queue已满
			if (myPool_isValid.front()) {
				unsigned short int* this_clip_id = myPool_id.front();
				for (int i = 0; i < step * step; i++) {
					boxExistTable[this_clip_id[i]] = true;
				}
			}
			delete[] myPool_cat.front();
			delete[] myPool_id.front();
			delete[] myPool_score.front();
			myPool_cat.pop_front();
			myPool_id.pop_front();
			myPool_score.pop_front();
			myPool_isValid.pop_front();
		}
		std::map<int, py::list> dict;
		for (auto iter = box_dict.begin(); iter != box_dict.end(); iter++) {
			if (iter->second.isExist && boxExistTable[iter->first]) {//&& boxExistTable[iter->first]
				py::list data;
				data.append(iter->second.x1);
				data.append(iter->second.y1);
				data.append(iter->second.x2);
				data.append(iter->second.y2);
				data.append(iter->second.label_id);
				data.append(iter->second.score);
				dict[iter->first] = data;
			}
		}
		return dict;
	}
	void setParams(int scale, int step, int h, int w) {
		img_scale = scale;
		this->step = step;
		img_h = h;
		img_w = w;
		margin = (img_scale - step) / 2;
		patch_cols = w / step;
		patch_rows = h / step;
		memcpy(findBoxTable, findBoxTable + MAX_BOX_NUM, 0);
		memcpy(boxExistTable, boxExistTable + MAX_BOX_NUM, 0);
	}
	const int& clamp(const int& lo, const int& hi, const int& v) {
		if (v < lo) return lo;
		if (v > hi)return hi;
		return v;
	}
public:
	std::map<int, boxInfo> box_dict;
	std::deque<unsigned short int*> myPool_cat;
	std::deque<unsigned short int*> myPool_id;
	std::deque<unsigned short int*> myPool_score;
	std::deque<bool> myPool_isValid;
public:
	int patch_cnt = 0;
	int box_cnt = 0;
	int img_scale;
	int step;
	int margin;
	int img_h;
	int img_w;
	int patch_rows;
	int patch_cols;
	bool findBoxTable[MAX_BOX_NUM];
	bool boxExistTable[MAX_BOX_NUM];
	int last_index = -1;
};
