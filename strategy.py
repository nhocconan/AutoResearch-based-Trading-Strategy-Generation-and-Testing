#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_Volume_Trend
Hypothesis: Price breaks above/below Camarilla R1/S1 levels on daily timeframe with volume spike and weekly EMA34 trend filter.
Targets 1d timeframe for lower trade frequency and better trend capture in both bull and bear markets.
Uses weekly EMA34 for trend direction to filter breakouts, reducing whipsaws.
Volume spike confirms institutional interest. Target: 15-25 trades/year to minimize fee drag.
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
    
    # Weekly EMA34 for trend filter (loaded once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily OHLC for Camarilla calculation (using previous day's values)
    # We'll calculate daily pivot from previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # handle first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla levels: based on previous day's range
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    range_prev = prev_high - prev_low
    r1 = prev_close + range_prev * 1.1 / 12
    s1 = prev_close - range_prev * 1.1 / 12
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and uptrend (price > weekly EMA34)
            if (price > r1[i] and vol_spike and price > ema34):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and downtrend (price < weekly EMA34)
            elif (price < s1[i] and vol_spike and price < ema34):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below weekly EMA34 OR breaks below S1 (reversal)
            if price < ema34 or price < s1[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above weekly EMA34 OR breaks above R1 (reversal)
            if price > ema34 or price > r1[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0