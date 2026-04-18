#!/usr/bin/env python3
"""
4h Camarilla Pivot Breakout with 1d Trend and Volume Filter
Hypothesis: Camarilla pivot levels (R1/S1) act as strong support/resistance.
Price breaking these levels with volume confirmation and aligned with higher
timeframe (1d) trend captures institutional breakout moves. Works in both
bull and bear markets by filtering counter-trend trades with 1d EMA.
Target: 20-30 trades/year to minimize fee drag while capturing strong moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Using daily OHLC from 1d data
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 4h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: current volume > 1.8x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema34_1d_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume, in uptrend
            if price > r1_val and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, in downtrend
            elif price < s1_val and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns below R1 or trend weakens
            if price < r1_val or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns above S1 or trend weakens
            if price > s1_val or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0