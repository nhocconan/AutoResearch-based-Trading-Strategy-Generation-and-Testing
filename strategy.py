#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Donchian and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day Donchian(20) breakout
    donch_high = np.full(len(close_1d), np.nan)
    donch_low = np.full(len(close_1d), np.nan)
    
    for i in range(19, len(close_1d)):
        donch_high[i] = np.max(high_1d[i-19:i+1])
        donch_low[i] = np.min(low_1d[i-19:i+1])
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # 1-day pivot points (standard)
    pivot = np.full(len(close_1d), np.nan)
    r1 = np.full(len(close_1d), np.nan)
    s1 = np.full(len(close_1d), np.nan)
    r2 = np.full(len(close_1d), np.nan)
    s2 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        pivot[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        r1[i] = 2 * pivot[i] - low_1d[i]
        s1[i] = 2 * pivot[i] - high_1d[i]
        r2[i] = pivot[i] + (high_1d[i] - low_1d[i])
        s2[i] = pivot[i] - (high_1d[i] - low_1d[i])
    
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 1-day volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):  # 20-period average
        vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # Donchian needs 20, volume needs 20
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below S1 or stoploss
            if (close[i] < s1_aligned[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above R1 or stoploss
            if (close[i] > r1_aligned[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with pivot filter and volume
            if volume_filter:
                # Long: break above Donchian high AND price above pivot (bullish bias)
                if (close[i] > donch_high_aligned[i] and 
                    close[i] > pivot_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: break below Donchian low AND price below pivot (bearish bias)
                elif (close[i] < donch_low_aligned[i] and 
                      close[i] < pivot_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Donchian and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day Donchian(20) breakout
    donch_high = np.full(len(close_1d), np.nan)
    donch_low = np.full(len(close_1d), np.nan)
    
    for i in range(19, len(close_1d)):
        donch_high[i] = np.max(high_1d[i-19:i+1])
        donch_low[i] = np.min(low_1d[i-19:i+1])
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # 1-day pivot points (standard)
    pivot = np.full(len(close_1d), np.nan)
    r1 = np.full(len(close_1d), np.nan)
    s1 = np.full(len(close_1d), np.nan)
    r2 = np.full(len(close_1d), np.nan)
    s2 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        pivot[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        r1[i] = 2 * pivot[i] - low_1d[i]
        s1[i] = 2 * pivot[i] - high_1d[i]
        r2[i] = pivot[i] + (high_1d[i] - low_1d[i])
        s2[i] = pivot[i] - (high_1d[i] - low_1d[i])
    
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 1-day volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):  # 20-period average
        vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # Donchian needs 20, volume needs 20
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below S1 or stoploss
            if (close[i] < s1_aligned[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above R1 or stoploss
            if (close[i] > r1_aligned[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with pivot filter and volume
            if volume_filter:
                # Long: break above Donchian high AND price above pivot (bullish bias)
                if (close[i] > donch_high_aligned[i] and 
                    close[i] > pivot_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: break below Donchian low AND price below pivot (bearish bias)
                elif (close[i] < donch_low_aligned[i] and 
                      close[i] < pivot_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals