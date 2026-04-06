#!/usr/bin/env python3
"""
6h Ichimoku Cloud with Daily Filter and Volume Confirmation
Hypothesis: Ichimoku signals (Tenkan/Kijun cross) filtered by daily cloud color (bullish/bearish) and volume spikes
capture trends while avoiding false signals in choppy markets. Works in bull/bear by only taking signals aligned
with higher timeframe trend (cloud color). Volume ensures momentum legitimacy.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_daily_filter_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components (9, 26, 52 periods)
    def _high_low_avg(arr, period):
        avg = np.full_like(arr, np.nan)
        for i in range(period - 1, len(arr)):
            avg[i] = (np.max(arr[i - period + 1:i + 1]) + np.min(arr[i - period + 1:i + 1])) / 2
        return avg
    
    tenkan = (_high_low_avg(high, 9) + _high_low_avg(low, 9)) / 2
    kijun = (_high_low_avg(high, 26) + _high_low_avg(low, 26)) / 2
    senkou_a = ((tenkan + kijun) / 2)
    senkou_b = (_high_low_avg(high, 52) + _high_low_avg(low, 52)) / 2
    
    # Daily trend filter: cloud color from 1D timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    tenkan_1d = (_high_low_avg(high_1d, 9) + _high_low_avg(low_1d, 9)) / 2
    kijun_1d = (_high_low_avg(high_1d, 26) + _high_low_avg(low_1d, 26)) / 2
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    senkou_b_1d = (_high_low_avg(high_1d, 52) + _high_low_avg(low_1d, 52)) / 2
    
    # Cloud bullish if Senkou A > Senkou B
    cloud_bullish_1d = senkou_a_1d > senkou_b_1d
    cloud_bullish_aligned = align_htf_to_ltf(prices, df_1d, cloud_bullish_1d)
    
    # Volume filter: current > 1.5x average of previous 24 periods
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i - 24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(52, 24)
    
    for i in range(start, n):
        # Skip if data not ready
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or \
           np.isnan(cloud_bullish_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.25 if position == 1 else -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Ichimoku signal: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
        ichimoku_bullish = tenkan[i] > kijun[i]
        ichimoku_bearish = tenkan[i] < kijun[i]
        
        # Volume condition
        vol_filter = volume[i] > vol_ma[i] * 1.5
        
        if position == 1:  # long
            # Exit: Ichimoku turns bearish OR price falls below cloud
            if ichimoku_bearish or close[i] < min(senkou_a[i], senkou_b[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short
            # Exit: Ichimoku turns bullish OR price rises above cloud
            if ichimoku_bullish or close[i] > max(senkou_a[i], senkou_b[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Entry conditions
            if ichimoku_bullish and cloud_bullish_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif ichimoku_bearish and not cloud_bullish_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Ichimoku Cloud with Daily Filter and Volume Confirmation
Hypothesis: Ichimoku signals (Tenkan/Kijun cross) filtered by daily cloud color (bullish/bearish) and volume spikes
capture trends while avoiding false signals in choppy markets. Works in bull/bear by only taking signals aligned
with higher timeframe trend (cloud color). Volume ensures momentum legitimacy.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_daily_filter_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components (9, 26, 52 periods)
    def _high_low_avg(arr, period):
        avg = np.full_like(arr, np.nan)
        for i in range(period - 1, len(arr)):
            avg[i] = (np.max(arr[i - period + 1:i + 1]) + np.min(arr[i - period + 1:i + 1])) / 2
        return avg
    
    tenkan = (_high_low_avg(high, 9) + _high_low_avg(low, 9)) / 2
    kijun = (_high_low_avg(high, 26) + _high_low_avg(low, 26)) / 2
    senkou_a = ((tenkan + kijun) / 2)
    senkou_b = (_high_low_avg(high, 52) + _high_low_avg(low, 52)) / 2
    
    # Daily trend filter: cloud color from 1D timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    tenkan_1d = (_high_low_avg(high_1d, 9) + _high_low_avg(low_1d, 9)) / 2
    kijun_1d = (_high_low_avg(high_1d, 26) + _high_low_avg(low_1d, 26)) / 2
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    senkou_b_1d = (_high_low_avg(high_1d, 52) + _high_low_avg(low_1d, 52)) / 2
    
    # Cloud bullish if Senkou A > Senkou B
    cloud_bullish_1d = senkou_a_1d > senkou_b_1d
    cloud_bullish_aligned = align_htf_to_ltf(prices, df_1d, cloud_bullish_1d)
    
    # Volume filter: current > 1.5x average of previous 24 periods
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i - 24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(52, 24)
    
    for i in range(start, n):
        # Skip if data not ready
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or \
           np.isnan(cloud_bullish_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.25 if position == 1 else -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Ichimoku signal: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
        ichimoku_bullish = tenkan[i] > kijun[i]
        ichimoku_bearish = tenkan[i] < kijun[i]
        
        # Volume condition
        vol_filter = volume[i] > vol_ma[i] * 1.5
        
        if position == 1:  # long
            # Exit: Ichimoku turns bearish OR price falls below cloud
            if ichimoku_bearish or close[i] < min(senkou_a[i], senkou_b[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short
            # Exit: Ichimoku turns bullish OR price rises above cloud
            if ichimoku_bullish or close[i] > max(senkou_a[i], senkou_b[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Entry conditions
            if ichimoku_bullish and cloud_bullish_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif ichimoku_bearish and not cloud_bullish_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</n>
</n>
</n>