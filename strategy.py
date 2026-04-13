#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation
    # Camarilla pivots identify intraday support/resistance where reversals often occur.
    # Volume spike confirms institutional interest at these levels.
    # Works in both bull/bear markets as pivots adapt to recent price action.
    # Discrete sizing (0.25) minimizes fee drag. Target: 12-37 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), R2 = C + ((H-L) * 1.1/6), R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12), S2 = C - ((H-L) * 1.1/6), S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Get 1d volume average for confirmation (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        idx_1d = i // (24 * 2)  # 1d bars in 12h timeframe (2 bars per day)
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 2.0 * vol_avg_20_1d_aligned[i]
        
        # Entry conditions: price touches Camarilla levels with volume
        # Long at S1/S2 with volume, Short at R1/R2 with volume
        price = close[i]
        touch_s1 = abs(price - s1_1d_aligned[i]) / s1_1d_aligned[i] < 0.001  # 0.1% tolerance
        touch_s2 = abs(price - s2_1d_aligned[i]) / s2_1d_aligned[i] < 0.001
        touch_r1 = abs(price - r1_1d_aligned[i]) / r1_1d_aligned[i] < 0.001
        touch_r2 = abs(price - r2_1d_aligned[i]) / r2_1d_aligned[i] < 0.001
        
        enter_long = (touch_s1 or touch_s2) and volume_confirmed
        enter_short = (touch_r1 or touch_r2) and volume_confirmed
        
        # Stoploss: 1.5x ATR based on 1d range
        if idx_1d < len(high_1d) and idx_1d < len(low_1d):
            daily_range = high_1d[idx_1d] - low_1d[idx_1d]
            stop_distance = daily_range * 0.75  # 75% of daily range
        else:
            stop_distance = 0
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0