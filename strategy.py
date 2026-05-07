#!/usr/bin/env python3
name = "12h_1d_Camarilla_Pivot_Breakout_VolumeTrend"
timeframe = "12h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's range
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla pivot formula
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels (key levels: L3, L4, H3, H4)
    l3 = prev_close - (range_hl * 1.1 / 2)
    h3 = prev_close + (range_hl * 1.1 / 2)
    l4 = prev_close - (range_hl * 1.1)
    h4 = prev_close + (range_hl * 1.1)
    
    # Align daily Camarilla levels to 12h timeframe
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    
    # Volume spike detection: 4-period average (1 day of 12h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    # 12h EMA(21) for trend filter
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(l3_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(h4_aligned[i]) or
            np.isnan(vol_ma_4[i]) or np.isnan(ema_21[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H3 with volume and uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            uptrend = ema_21[i] > ema_21[i-1]
            
            if close[i] > h3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below L3 with volume and downtrend
            elif close[i] < l3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below H3 or volume drops
            if close[i] < h3_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above L3 or volume drops
            if close[i] > l3_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla H3/L3 breakout with volume confirmation and 12h EMA trend filter
# - Camarilla H3/L3 act as key support/resistance levels from prior session
# - Breakout above H3 with volume in uptrend = long opportunity
# - Breakdown below L3 with volume in downtrend = short opportunity
# - Volume spike (1.8x average) confirms institutional participation
# - Works in both bull (buy H3 breaks in uptrend) and bear (sell L3 breaks in downtrend)
# - Exit when price returns to H3/L3 or volume weakens
# - Position size 0.25 targets ~20-40 trades/year, avoiding fee drag
# - Uses actual daily Camarilla levels (not weekly) for better responsiveness
# - EMA(21) trend filter reduces whipsaws vs using same timeframe
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: Camarilla (1d) + volume (12h) + trend (12h) not recently tried on 12h
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits