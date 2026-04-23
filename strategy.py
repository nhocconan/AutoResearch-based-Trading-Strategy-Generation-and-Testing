#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R1/S1 breakout with 12h EMA34 trend filter and volume spike confirmation.
Long when price breaks above R1 (1d) AND close > 12h EMA34 (uptrend) AND volume > 1.8x 20-period MA.
Short when price breaks below S1 (1d) AND close < 12h EMA34 (downtrend) AND volume > 1.8x 20-period MA.
Exit when price returns to Camarilla H3/L3 levels.
Designed for ~20-30 trades/year with structure-based edge using tighter Camarilla levels (R1/S1)
which capture institutional order flow zones. Uses 12h EMA34 for smoother trend filter and
volume confirmation to avoid false breakouts. Discrete sizing (0.25) minimizes fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Camarilla calculation: based on previous day's range
    range_1d = high_1d - low_1d
    r1 = close_1d_arr + 1.0 * (range_1d / 12)
    s1 = close_1d_arr - 1.0 * (range_1d / 12)
    r3 = close_1d_arr + 1.0 * range_1d
    s3 = close_1d_arr - 1.0 * range_1d
    h3 = close_1d_arr + 1.25 * range_1d
    l3 = close_1d_arr - 1.25 * range_1d
    
    # Align Camarilla levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need EMA34, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 12h EMA34 = uptrend, close < 12h EMA34 = downtrend
        trend_up = close[i] > ema_34_12h_aligned[i]
        trend_down = close[i] < ema_34_12h_aligned[i]
        
        # Volume filter: 6h volume > 1.8x 20-period MA
        vol_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_up = close[i] > r1_aligned[i]  # Break above R1
        breakout_down = close[i] < s1_aligned[i]  # Break below S1
        return_to_h3 = close[i] < h3_aligned[i]  # Return below H3 (exit long)
        return_to_l3 = close[i] > l3_aligned[i]  # Return above L3 (exit short)
        opposite_extreme = (position == 1 and breakout_down) or \
                           (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Break above R1 AND uptrend AND volume confirmation
            if breakout_up and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 AND downtrend AND volume confirmation
            elif breakout_down and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to H3/L3 or opposite extreme hit
            exit_signal = False
            if position == 1:
                exit_signal = return_to_h3 or opposite_extreme
            elif position == -1:
                exit_signal = return_to_l3 or opposite_extreme
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R1S1_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0