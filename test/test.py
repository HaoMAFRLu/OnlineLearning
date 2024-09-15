mode = 'newton'
is_shift_dis = True
is_clear     = False
is_reset     = False

name1 = 'w_shift' if is_shift_dis is True else 'wo_shift'
name2 = 'w_clear' if is_clear is True else 'wo_clear'
name3 = 'w_reset' if is_reset is True else 'wo_reset'
root_name = mode + '_' + name1 + '_' + name2 + '_' + name3

print(root_name)