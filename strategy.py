#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dEMA34_v1
Hypothesis: Trade 1h Camarilla R1/S1 breakouts with 4h EMA34 trend filter and 1d EMA34 trend confirmation.
R1/S1 levels provide frequent but valid breakout opportunities when aligned with higher timeframe trends.
In bull markets: price breaks above R1 with 4h and 1d uptrend → long continuation.
In bear markets: price breaks below S1 with 4h and 1d downtrend → short continuation.
Volume confirmation ensures breakouts have participation.
Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:
        return np.zeros(n)
    
    # Get 1d data for additional trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA(34) on 4h for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate EMA(34) on 1d for trend confirmation
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 4h bar
    # Using R1/S1 for breakout entries (more frequent but still significant levels)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Where C = (H+L+Close)/3 of previous 4h bar
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_4h['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_4h['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_4h['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)
    s1 = pivot - (range_hl * 1.1 / 12.0)
    
    # Align Camarilla levels to 1h
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume confirmation: current volume > 1.5 * 24-period average (6h average on 1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 4h EMA(34), 1d EMA(34), volume MA(24), and need 4h data
    start_idx = max(34, 24) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        trend_4h_up = close_val > ema_34_4h_aligned[i]   # 4h uptrend
        trend_4h_down = close_val < ema_34_4h_aligned[i]  # 4h downtrend
        trend_1d_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_1d_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirm AND 4h uptrend AND 1d uptrend
            long_signal = (close_val > r1_aligned[i]) and vol_conf and trend_4h_up and trend_1d_up
            
            # Short: price breaks below S1 AND volume confirm AND 4h downtrend AND 1d downtrend
            short_signal = (close_val < s1_aligned[i]) and vol_conf and trend_4h_down and trend_1d_down
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price drops below S1 (failed breakout) OR 4h trend flips down OR 1d trend flips down
            if (close_val < s1_aligned[i]) or (not trend_4h_up) or (not trend_1d_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price rises above R1 (failed breakdown) OR 4h trend flips up OR 1d trend flips up
            if (close_val > r1_aligned[i]) or (not trend_4h_down) or (not trend_1d_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dEMA34_v1"
timeframe = "1h"
leverage = 1.0