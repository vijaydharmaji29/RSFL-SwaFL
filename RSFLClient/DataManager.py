import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import random
import math


class DataManager:
    def __init__(self, datasetSize):
        _ratings = tfds.load("movielens/100k-ratings", split="train")
        self.ratings = _ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
            "user_rating": x["user_rating"]
        })
        self.datasetSize = datasetSize

        # get movie title to make embedding
        movies = self.ratings.map(lambda x: x["movie_title"])
        self.movie_titles = np.unique(np.concatenate(list(movies.batch(100_000))))

        # get userID to make user embedding
        userIDs = self.ratings.map(lambda x: x["user_id"])
        self.userIDs = np.unique(np.concatenate(list(userIDs.batch(100_000))))

        print(f"Dataset have total unique {len(self.userIDs)} users and {len(self.movie_titles)} movies")

        self.cluster_users = [[38, 42, 43, 56, 83, 87, 109, 119, 125, 127, 137, 141, 152, 159, 164, 178, 230, 301, 311, 314, 320, 346, 348, 357, 373, 374, 388, 393, 459, 484, 487, 493, 506, 507, 513, 518, 533, 534, 536, 545, 548, 577, 588, 620, 642, 644, 648, 654, 671, 677, 689, 705, 712, 714, 721, 749, 759, 764, 796, 798, 804, 807, 825, 826, 850, 881, 882, 887, 897, 901, 927, 939], [2, 4, 8, 9, 11, 12, 14, 19, 22, 24, 25, 27, 28, 29, 30, 33, 34, 35, 36, 37, 39, 41, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 57, 60, 61, 65, 66, 67, 70, 74, 78, 79, 80, 81, 84, 86, 88, 89, 91, 93, 95, 96, 97, 98, 99, 100, 101, 103, 105, 106, 110, 111, 112, 113, 114, 117, 120, 126, 128, 131, 132, 134, 135, 136, 138, 139, 140, 142, 143, 144, 147, 148, 150, 156, 157, 158, 160, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 179, 180, 182, 185, 186, 187, 188, 190, 191, 195, 196, 197, 203, 208, 209, 212, 215, 216, 218, 219, 220, 222, 223, 224, 225, 226, 228, 231, 237, 238, 240, 241, 242, 245, 247, 251, 252, 253, 254, 257, 258, 259, 260, 261, 265, 266, 272, 273, 274, 277, 278, 281, 282, 283, 284, 285, 286, 287, 288, 290, 294, 298, 300, 304, 306, 309, 310, 313, 317, 318, 319, 321, 322, 324, 328, 331, 333, 335, 336, 337, 340, 341, 347, 350, 351, 352, 353, 355, 356, 358, 359, 362, 364, 365, 366, 367, 368, 369, 370, 371, 372, 375, 376, 377, 384, 386, 390, 395, 396, 398, 400, 402, 403, 404, 406, 408, 411, 412, 413, 414, 415, 419, 420, 421, 424, 426, 427, 428, 431, 432, 434, 435, 436, 438, 439, 440, 441, 443, 444, 447, 451, 453, 455, 462, 464, 466, 469, 470, 471, 473, 476, 477, 481, 482, 489, 491, 492, 494, 499, 501, 502, 504, 510, 511, 512, 514, 515, 516, 517, 519, 520, 522, 523, 525, 527, 528, 529, 530, 531, 535, 539, 540, 541, 546, 547, 549, 550, 553, 554, 555, 556, 557, 558, 559, 562, 563, 564, 565, 568, 571, 573, 574, 576, 579, 580, 581, 583, 584, 585, 589, 593, 594, 596, 597, 598, 599, 600, 602, 603, 604, 611, 612, 613, 614, 616, 618, 619, 621, 623, 624, 628, 630, 631, 632, 633, 634, 636, 638, 641, 647, 649, 658, 659, 661, 662, 663, 667, 668, 670, 672, 673, 674, 675, 676, 678, 679, 680, 681, 684, 687, 688, 690, 691, 692, 696, 697, 700, 701, 703, 706, 708, 709, 715, 717, 718, 720, 722, 723, 725, 726, 728, 729, 731, 732, 734, 735, 738, 739, 740, 743, 744, 746, 748, 750, 754, 755, 761, 767, 768, 769, 770, 771, 772, 775, 776, 777, 779, 780, 782, 783, 784, 785, 786, 787, 791, 793, 794, 799, 800, 801, 802, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 821, 822, 827, 830, 832, 834, 835, 837, 838, 839, 841, 842, 845, 849, 851, 852, 855, 856, 857, 859, 861, 863, 867, 871, 872, 873, 874, 876, 877, 879, 885, 886, 888, 891, 893, 895, 899, 904, 906, 909, 911, 912, 914, 919, 921, 922, 923, 925, 928, 929, 931, 934, 935, 936, 938, 940, 941, 942, 943],
                              [5, 20, 23, 77, 82, 153, 161, 198, 202, 207, 236, 246, 262, 268, 275, 279, 299, 305, 316, 325, 326, 327, 354, 363, 380, 387, 401, 425, 442, 452, 454, 456, 465, 483, 488, 496, 497, 498, 500, 505, 521, 524, 566, 567, 586, 601, 617, 625, 627, 639, 650, 665, 666, 682, 693, 698, 699, 707, 727, 745, 756, 766, 773, 778, 788, 790, 795, 805, 847, 865, 878, 889, 896, 908, 918, 930],
                              [405],
                              [13],
                              [3, 6, 15, 17, 18, 21, 26, 31, 32, 40, 58, 62, 63, 64, 68, 69, 71, 72, 73, 75, 76, 85, 92, 104, 107, 108, 115, 116, 121, 122, 123, 124, 129, 133, 146, 149, 154, 155, 176, 177, 183, 184, 192, 193, 199, 204, 205, 206, 211, 214, 217, 221, 227, 229, 232, 235, 239, 243, 244, 248, 250, 255, 263, 271, 280, 289, 297, 302, 307, 308, 315, 323, 329, 334, 338, 342, 344, 345, 349, 360, 361, 378, 381, 382, 383, 389, 391, 392, 397, 407, 409, 410, 417, 418, 422, 423, 429, 430, 433, 437, 446, 448, 449, 458, 460, 461, 463, 467, 475, 478, 479, 480, 485, 486, 490, 508, 509, 526, 538, 542, 543, 544, 552, 560, 569, 570, 572, 575, 578, 582, 587, 590, 591, 595, 605, 607, 608, 609, 610, 615, 622, 626, 635, 637, 643, 645, 646, 651, 652, 656, 657, 664, 669, 683, 685, 695, 702, 704, 710, 711, 713, 719, 724, 730, 733, 736, 737, 741, 742, 751, 752, 753, 757, 760, 762, 763, 765, 781, 789, 792, 797, 803, 806, 820, 824, 828, 829, 831, 836, 840, 844, 853, 854, 858, 860, 866, 869, 870, 875, 883, 884, 890, 894, 898, 900, 902, 903, 905, 910, 913, 915, 916, 917, 920, 924, 926, 932, 937],
                              [49, 102, 194, 201, 234, 269, 293, 385, 399, 561, 653, 660, 774, 833, 843, 868, 933],
                              [130, 200, 256, 330, 332, 416, 450, 472, 532, 551, 907],
                              [181, 445, 537, 655],
                              [1, 7, 10, 16, 59, 90, 94, 118, 145, 151, 189, 210, 213, 233, 249, 264, 267, 270, 276, 291, 292, 295, 296, 303, 312, 339, 343, 379, 394, 457, 468, 474, 495, 503, 592, 606, 629, 640, 686, 694, 716, 747, 758, 823, 846, 848, 862, 864, 880, 892]]

        self.user_clusters = {42: 0, 43: 0, 56: 0, 83: 0, 109: 0, 119: 0, 125: 0, 127: 0, 137: 0, 141: 0, 159: 0, 164: 0, 168: 0, 178: 0, 197: 0, 230: 0, 301: 0, 311: 0, 320: 0, 346: 0, 347: 0, 348: 0, 357: 0, 373: 0, 374: 0, 388: 0, 393: 0, 396: 0, 403: 0, 451: 0, 459: 0, 484: 0, 487: 0, 489: 0, 493: 0, 504: 0, 506: 0, 507: 0, 513: 0, 518: 0, 533: 0, 534: 0, 536: 0, 545: 0, 548: 0, 577: 0, 588: 0, 619: 0, 620: 0, 644: 0, 648: 0, 654: 0, 671: 0, 677: 0, 689: 0, 705: 0, 712: 0, 714: 0, 718: 0, 721: 0, 749: 0, 759: 0, 764: 0, 798: 0, 804: 0, 807: 0, 825: 0, 826: 0, 850: 0, 881: 0, 882: 0, 901: 0, 927: 0, 939: 0, 4: 1, 8: 1, 9: 1, 12: 1, 25: 1, 27: 1, 28: 1, 29: 1, 30: 1, 33: 1, 35: 1, 36: 1, 37: 1, 39: 1, 44: 1, 45: 1, 46: 1, 48: 1, 50: 1, 51: 1, 52: 1, 53: 1, 54: 1, 57: 1, 65: 1, 66: 1, 67: 1, 70: 1, 74: 1, 79: 1, 80: 1, 84: 1, 88: 1, 89: 1, 91: 1, 93: 1, 95: 1, 96: 1, 97: 1, 98: 1, 99: 1, 100: 1, 103: 1, 106: 1, 110: 1, 111: 1, 112: 1, 113: 1, 114: 1, 117: 1, 120: 1, 126: 1, 128: 1, 131: 1, 134: 1, 135: 1, 138: 1, 139: 1, 140: 1, 143: 1, 147: 1, 148: 1, 150: 1, 157: 1, 160: 1, 162: 1, 165: 1, 166: 1, 167: 1, 169: 1, 170: 1, 172: 1, 173: 1, 174: 1, 175: 1, 179: 1, 180: 1, 182: 1, 185: 1, 186: 1, 187: 1, 188: 1, 190: 1, 191: 1, 195: 1, 196: 1, 203: 1, 208: 1, 212: 1, 216: 1, 220: 1, 222: 1, 224: 1, 225: 1, 226: 1, 228: 1, 231: 1, 237: 1, 238: 1, 240: 1, 241: 1, 242: 1, 245: 1, 247: 1, 251: 1, 252: 1, 253: 1, 254: 1, 257: 1, 258: 1, 259: 1, 260: 1, 261: 1, 265: 1, 272: 1, 274: 1, 278: 1, 281: 1, 282: 1, 283: 1, 285: 1, 286: 1, 287: 1, 290: 1, 294: 1, 298: 1, 300: 1, 304: 1, 306: 1, 309: 1, 313: 1, 317: 1, 318: 1, 319: 1, 324: 1, 328: 1, 333: 1, 335: 1, 336: 1, 337: 1, 340: 1, 341: 1, 350: 1, 351: 1, 352: 1, 355: 1, 356: 1, 359: 1, 362: 1, 364: 1, 366: 1, 367: 1, 368: 1, 369: 1, 371: 1, 372: 1, 375: 1, 377: 1, 384: 1, 386: 1, 390: 1, 395: 1, 398: 1, 400: 1, 402: 1, 404: 1, 406: 1, 408: 1, 411: 1, 412: 1, 414: 1, 415: 1, 419: 1, 421: 1, 426: 1, 427: 1, 428: 1, 431: 1, 432: 1, 434: 1, 435: 1, 436: 1, 438: 1, 440: 1, 441: 1, 443: 1, 444: 1, 447: 1, 453: 1, 462: 1, 464: 1, 466: 1, 469: 1, 470: 1, 471: 1, 473: 1, 476: 1, 477: 1, 481: 1, 482: 1, 492: 1, 494: 1, 497: 1, 499: 1, 501: 1, 502: 1, 510: 1, 511: 1, 512: 1, 514: 1, 516: 1, 520: 1, 523: 1, 528: 1, 529: 1, 530: 1, 531: 1, 539: 1, 540: 1, 541: 1, 546: 1, 549: 1, 550: 1, 553: 1, 554: 1, 555: 1, 557: 1, 558: 1, 562: 1, 563: 1, 564: 1, 565: 1, 568: 1, 571: 1, 574: 1, 576: 1, 579: 1, 580: 1, 583: 1, 584: 1, 585: 1, 586: 1, 589: 1, 593: 1, 596: 1, 597: 1, 598: 1, 599: 1, 600: 1, 602: 1, 603: 1, 604: 1, 611: 1, 612: 1, 613: 1, 616: 1, 618: 1, 621: 1, 623: 1, 624: 1, 628: 1, 630: 1, 632: 1, 634: 1, 636: 1, 638: 1, 647: 1, 649: 1, 659: 1, 661: 1, 663: 1, 668: 1, 670: 1, 673: 1, 674: 1, 676: 1, 680: 1, 684: 1, 687: 1, 688: 1, 690: 1, 691: 1, 692: 1, 697: 1, 700: 1, 701: 1, 703: 1, 706: 1, 708: 1, 709: 1, 715: 1, 717: 1, 720: 1, 722: 1, 723: 1, 725: 1, 728: 1, 732: 1, 738: 1, 739: 1, 740: 1, 746: 1, 748: 1, 750: 1, 755: 1, 761: 1, 767: 1, 768: 1, 769: 1, 770: 1, 771: 1, 772: 1, 775: 1, 776: 1, 777: 1, 779: 1, 780: 1, 782: 1, 783: 1, 784: 1, 785: 1, 786: 1, 787: 1, 788: 1, 791: 1, 793: 1, 799: 1, 800: 1, 801: 1, 802: 1, 808: 1, 809: 1, 810: 1, 811: 1, 812: 1, 814: 1, 815: 1, 816: 1, 817: 1, 818: 1, 819: 1, 821: 1, 822: 1, 830: 1, 834: 1, 835: 1, 838: 1, 841: 1, 845: 1, 849: 1, 851: 1, 852: 1, 855: 1, 856: 1, 859: 1, 861: 1, 863: 1, 867: 1, 871: 1, 872: 1, 873: 1, 876: 1, 879: 1, 885: 1, 886: 1, 888: 1, 891: 1, 893: 1, 895: 1, 899: 1, 904: 1, 906: 1, 909: 1, 919: 1, 921: 1, 922: 1, 923: 1, 925: 1, 928: 1, 931: 1, 934: 1, 935: 1, 938: 1, 940: 1, 941: 1, 942: 1, 943: 1, 5: 2, 20: 2, 23: 2, 82: 2, 153: 2, 161: 2, 198: 2, 207: 2, 236: 2, 246: 2, 262: 2, 268: 2, 275: 2, 279: 2, 299: 2, 305: 2, 316: 2, 325: 2, 326: 2, 327: 2, 354: 2, 363: 2, 380: 2, 387: 2, 401: 2, 425: 2, 442: 2, 445: 2, 452: 2, 454: 2, 456: 2, 465: 2, 488: 2, 496: 2, 498: 2, 500: 2, 505: 2, 521: 2, 524: 2, 566: 2, 567: 2, 601: 2, 617: 2, 627: 2, 637: 2, 639: 2, 650: 2, 666: 2, 682: 2, 693: 2, 698: 2, 699: 2, 707: 2, 727: 2, 745: 2, 756: 2, 766: 2, 773: 2, 778: 2, 790: 2, 795: 2, 805: 2, 847: 2, 854: 2, 865: 2, 878: 2, 889: 2, 896: 2, 916: 2, 918: 2, 930: 2, 405: 3, 13: 4, 49: 5, 102: 5, 194: 5, 201: 5, 234: 5, 269: 5, 293: 5, 385: 5, 399: 5, 537: 5, 561: 5, 653: 5, 655: 5, 660: 5, 774: 5, 833: 5, 843: 5, 868: 5, 933: 5, 2: 6, 3: 6, 6: 6, 11: 6, 14: 6, 15: 6, 17: 6, 18: 6, 19: 6, 21: 6, 22: 6, 26: 6, 31: 6, 32: 6, 34: 6, 40: 6, 41: 6, 47: 6, 55: 6, 58: 6, 61: 6, 62: 6, 63: 6, 64: 6, 68: 6, 69: 6, 71: 6, 72: 6, 73: 6, 75: 6, 76: 6, 77: 6, 78: 6, 81: 6, 85: 6, 86: 6, 92: 6, 101: 6, 104: 6, 105: 6, 107: 6, 108: 6, 115: 6, 116: 6, 121: 6, 122: 6, 123: 6, 124: 6, 129: 6, 132: 6, 133: 6, 136: 6, 142: 6, 144: 6, 146: 6, 149: 6, 154: 6, 155: 6, 156: 6, 158: 6, 163: 6, 171: 6, 176: 6, 177: 6, 183: 6, 184: 6, 192: 6, 193: 6, 199: 6, 202: 6, 204: 6, 205: 6, 206: 6, 209: 6, 211: 6, 214: 6, 215: 6, 217: 6, 218: 6, 219: 6, 221: 6, 223: 6, 227: 6, 229: 6, 232: 6, 235: 6, 239: 6, 243: 6, 244: 6, 248: 6, 250: 6, 255: 6, 263: 6, 266: 6, 271: 6, 273: 6, 277: 6, 280: 6, 284: 6, 288: 6, 289: 6, 297: 6, 302: 6, 307: 6, 308: 6, 310: 6, 315: 6, 321: 6, 322: 6, 323: 6, 329: 6, 331: 6, 334: 6, 338: 6, 342: 6, 344: 6, 345: 6, 349: 6, 353: 6, 358: 6, 360: 6, 361: 6, 365: 6, 370: 6, 376: 6, 378: 6, 381: 6, 382: 6, 383: 6, 389: 6, 391: 6, 392: 6, 397: 6, 407: 6, 409: 6, 410: 6, 413: 6, 417: 6, 418: 6, 420: 6, 422: 6, 423: 6, 424: 6, 429: 6, 430: 6, 433: 6, 437: 6, 439: 6, 446: 6, 448: 6, 449: 6, 455: 6, 458: 6, 460: 6, 461: 6, 463: 6, 467: 6, 475: 6, 478: 6, 479: 6, 480: 6, 483: 6, 485: 6, 486: 6, 490: 6, 491: 6, 508: 6, 509: 6, 515: 6, 517: 6, 519: 6, 522: 6, 525: 6, 526: 6, 527: 6, 535: 6, 538: 6, 542: 6, 543: 6, 544: 6, 547: 6, 552: 6, 556: 6, 559: 6, 560: 6, 569: 6, 570: 6, 572: 6, 573: 6, 575: 6, 578: 6, 581: 6, 582: 6, 587: 6, 590: 6, 591: 6, 594: 6, 595: 6, 605: 6, 607: 6, 608: 6, 609: 6, 610: 6, 614: 6, 615: 6, 622: 6, 625: 6, 626: 6, 631: 6, 633: 6, 635: 6, 641: 6, 643: 6, 645: 6, 646: 6, 651: 6, 652: 6, 656: 6, 657: 6, 658: 6, 662: 6, 664: 6, 665: 6, 667: 6, 669: 6, 672: 6, 675: 6, 678: 6, 679: 6, 681: 6, 683: 6, 685: 6, 695: 6, 696: 6, 702: 6, 704: 6, 710: 6, 711: 6, 713: 6, 719: 6, 724: 6, 726: 6, 729: 6, 730: 6, 731: 6, 733: 6, 734: 6, 735: 6, 736: 6, 737: 6, 741: 6, 742: 6, 743: 6, 744: 6, 751: 6, 752: 6, 753: 6, 754: 6, 757: 6, 760: 6, 762: 6, 763: 6, 765: 6, 781: 6, 789: 6, 792: 6, 794: 6, 797: 6, 803: 6, 806: 6, 813: 6, 820: 6, 824: 6, 827: 6, 828: 6, 829: 6, 831: 6, 832: 6, 836: 6, 837: 6, 839: 6, 842: 6, 844: 6, 853: 6, 857: 6, 858: 6, 860: 6, 866: 6, 869: 6, 870: 6, 874: 6, 875: 6, 877: 6, 883: 6, 884: 6, 890: 6, 894: 6, 898: 6, 900: 6, 902: 6, 903: 6, 905: 6, 908: 6, 910: 6, 911: 6, 912: 6, 913: 6, 914: 6, 915: 6, 917: 6, 920: 6, 924: 6, 926: 6, 929: 6, 932: 6, 936: 6, 937: 6, 1: 7, 7: 7, 10: 7, 16: 7, 24: 7, 59: 7, 60: 7, 90: 7, 94: 7, 118: 7, 145: 7, 151: 7, 189: 7, 210: 7, 213: 7, 233: 7, 249: 7, 264: 7, 267: 7, 270: 7, 276: 7, 291: 7, 292: 7, 295: 7, 296: 7, 303: 7, 312: 7, 339: 7, 343: 7, 379: 7, 394: 7, 457: 7, 468: 7, 474: 7, 495: 7, 503: 7, 592: 7, 606: 7, 629: 7, 640: 7, 686: 7, 694: 7, 716: 7, 747: 7, 758: 7, 823: 7, 840: 7, 846: 7, 848: 7, 862: 7, 864: 7, 880: 7, 892: 7, 38: 8, 87: 8, 130: 8, 152: 8, 200: 8, 256: 8, 314: 8, 330: 8, 332: 8, 416: 8, 450: 8, 472: 8, 532: 8, 551: 8, 642: 8, 796: 8, 887: 8, 897: 8, 907: 8, 181: 9}
    
    def generateSkewedDataset(self, number_of_labels): #returns tensor dataset to train on
        tensorSlices = {"movie_title": [], "user_id": [], "user_rating": []}

        #accuracy of each cluster is known
        #sort it out based on least accurate to most accurate
        #proportionally assign cluster size for dataset ~ datasetSize

        sorted_accuracy = [(1, i) for i in range(number_of_labels)]
        random.shuffle(sorted_accuracy)        

        cluster_sizes = [0]*number_of_labels

        n = 10000

        for k in range(1, 11):
            cluster = sorted_accuracy[k - 1][1]
            max_size = (n/55)*k
            cluster_sizes[cluster] = max_size
        
        shuffled_ratings = list(self.ratings)
        random.shuffle(shuffled_ratings)

        current_count = [0]*number_of_labels

        for i in shuffled_ratings:

            movieTitle = i["movie_title"]
            userId = i["user_id"]
            userRating = i["user_rating"]

            cluster_number = self.user_clusters[int(userId.numpy().decode('utf-8'))]

            if(current_count[cluster_number] < cluster_sizes[cluster_number]):
                tensorSlices["movie_title"].append(movieTitle)
                tensorSlices["user_id"].append(userId)
                tensorSlices["user_rating"].append(userRating)
                current_count[cluster_number] += 1


        skewedRatings = tf.data.Dataset.from_tensor_slices(tensorSlices)
        return skewedRatings, cluster_sizes


    def calculateBhattacharyaDistance(self, p, q):
        np = sum(p)
        p_new = [x/np for x in p]
        nq = sum(q)
        q_new = [x/nq for x in q]

        bc = 0

        for i in range(len(p_new)):
            bc += math.sqrt(p_new[i] * q_new[i])

        bd = -math.log(bc)

        return bd
