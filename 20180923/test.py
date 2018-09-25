import utils
class_counter, class_weight_sum, class_average, total_average = utils.load_gate_csv()
print(class_counter)
print(class_weight_sum)
print(class_average)
print(total_average)


# result
# {1.0: 3, 4.0: 2, 7.0: 2, 10.0: 3}
# {1.0: tensor([9., 7.]), 4.0: tensor([7., 7.]), 7.0: tensor([11., 13.]), 10.0: tensor([16., 17.])}
# {1.0: tensor([3.0000, 2.3333]), 4.0: tensor([3.5000, 3.5000]), 7.0: tensor([5.5000, 6.5000]), 10.0: tensor([5.3333, 5.6667])}
# tensor([4.3000, 4.4000])