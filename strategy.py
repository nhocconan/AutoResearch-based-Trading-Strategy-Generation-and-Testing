#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla long/short levels from 1d high/low/close + volume confirmation + 1d trend filter
# Designed for low trade frequency (target 12-37/year) with clear mean reversion logic
# Works in both bull (buy dips) and bear (sell rallies) markets by fading extremes
# Uses volume spike to confirm institutional interest at pivot levels

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # H4 = close + 1.1/12 * (high - low)  # Long entry (sell rally)
    # L4 = close - 1.1/12 * (high - low)  # Short entry (buy dip)
    rng = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1/12 * rng
    camarilla_l4 = close_1d - 1.1/12 * rng
    
    # 1d EMA20 for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price touches L4 (buy dip) + above trend + volume spike
        if (close[i] <= camarilla_l4_aligned[i] and 
            close[i] > ema20_1d_aligned[i] and 
            volume[i] > 1.5 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price touches H4 (sell rally) + below trend + volume spike
        elif (close[i] >= camarilla_h4_aligned[i] and 
              close[i] < ema20_1d_aligned[i] and 
              volume[i] > 1.5 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price crosses EMA20
        elif position == 1 and close[i] < ema20_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema20_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_1dVolume_1dTrend_MeanReversion"
timeframe = "12h"
leverage = 1.0