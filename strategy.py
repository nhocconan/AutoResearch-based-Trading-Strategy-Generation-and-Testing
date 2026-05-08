#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot level touch with 12h EMA trend filter and volume confirmation.
# Long when price touches Camarilla S1 (buy the dip) AND 12h EMA50 rising AND volume > 1.5x 20-period average.
# Short when price touches Camarilla R1 (sell the rip) AND 12h EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price closes above/below previous bar's close or on opposite touch.
# Camarilla levels provide high-probability reversal zones in ranging markets. EMA filter ensures trend alignment.
# Volume spike confirms institutional participation. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Camarilla_S1R1_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: H, L, C from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    # S1 = C - (H-L)*1.06/12
    # R1 = C + (H-L)*1.06/12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.06 / 12
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.06 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h EMA50 direction
    ema50_rising = np.zeros_like(ema50_12h_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_12h_aligned, dtype=bool)
    ema50_rising[1:] = ema50_12h_aligned[1:] > ema50_12h_aligned[:-1]
    ema50_falling[1:] = ema50_12h_aligned[1:] < ema50_12h_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Check for NaN values
        if (np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price touches Camarilla S1 (within 0.1%), 12h EMA50 rising, volume filter
            long_cond = (abs(low[i] - camarilla_s1_aligned[i]) / camarilla_s1_aligned[i] < 0.001) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price touches Camarilla R1 (within 0.1%), 12h EMA50 falling, volume filter
            short_cond = (abs(high[i] - camarilla_r1_aligned[i]) / camarilla_r1_aligned[i] < 0.001) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes above previous bar's close or touches R1
            if close[i] > close[i-1] or abs(high[i] - camarilla_r1_aligned[i]) / camarilla_r1_aligned[i] < 0.001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes below previous bar's close or touches S1
            if close[i] < close[i-1] or abs(low[i] - camarilla_s1_aligned[i]) / camarilla_s1_aligned[i] < 0.001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals