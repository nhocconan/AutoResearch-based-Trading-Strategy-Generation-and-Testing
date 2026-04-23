#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R1 AND 12h EMA50 rising AND volume > 2x 20-period average.
Short when price breaks below Camarilla S1 AND 12h EMA50 falling AND volume > 2x 20-period average.
Exit when price touches the opposite Camarilla level (S1 for longs, R1 for shorts).
Uses 12h HTF for EMA trend direction (avoids whipsaws in ranging markets). Target: 75-200 total trades over 4 years (19-50/year).
Camarilla pivot levels provide precise intraday support/resistance; EMA50 filter ensures we trade with the intermediate-term trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Camarilla pivot levels (based on previous day's OHLC)
    # We need daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar using prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 levels: R1 = Close + 1.1*(High-Low)/12, S1 = Close - 1.1*(High-Low)/12
    camarilla_r1_1d = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1_1d = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align daily Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d, additional_delay_bars=1)
    camarilla_s1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d, additional_delay_bars=1)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # volume MA (20), EMA50 (50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r1_1d_aligned[i]) or 
            np.isnan(camarilla_s1_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = camarilla_r1_1d_aligned[i]
        s1 = camarilla_s1_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= 1:
            ema_prev = ema_50_12h_aligned[i-1]
            ema_curr = ema_50_12h_aligned[i]
            ema_rising = ema_curr > ema_prev
            ema_falling = ema_curr < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Camarilla R1 AND EMA50 rising AND volume spike
            if price > r1 and ema_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 AND EMA50 falling AND volume spike
            elif price < s1 and ema_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price touches opposite Camarilla level
            if position == 1 and price < s1:  # Long exit at Camarilla S1
                exit_signal = True
            elif position == -1 and price > r1:  # Short exit at Camarilla R1
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeSpike_LevelExit"
timeframe = "4h"
leverage = 1.0