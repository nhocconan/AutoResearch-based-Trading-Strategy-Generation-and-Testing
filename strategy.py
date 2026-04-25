#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1wTrend_Filter_VolumeSpike
Hypothesis: 12h Camarilla H3/L3 breakout with 1w trend filter (price >/< EMA50) and volume confirmation (>1.8x 20-bar avg). 
Enters long when price breaks above H3 in 1w uptrend with volume spike, short when breaks below L3 in 1w downtrend with volume spike. 
Exits on opposite Camarilla level touch (L3 for longs, H3 for shorts) or trend reversal. 
Designed for 12h timeframe with ~12-37 trades/year, works in bull/bear by following 1w trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels for 12h timeframe using previous bar's OHLC
    # Camarilla: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    #            H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    h3 = prev_close + 1.1 * camarilla_range
    l3 = prev_close - 1.1 * camarilla_range
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need at least 1 bar of previous data and EMA50 warmup
    start_idx = max(50, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(h3[i]) or 
            np.isnan(l3[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 in 1w uptrend with volume confirmation
            long_setup = (close[i] > h3[i]) and (close[i] > ema_50_1w_aligned[i]) and volume_spike[i]
            # Short: price breaks below L3 in 1w downtrend with volume confirmation
            short_setup = (close[i] < l3[i]) and (close[i] < ema_50_1w_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.30
                position = 1
            elif short_setup:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price touches L3 OR trend turns down
            if (close[i] <= l3[i]) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price touches H3 OR trend turns up
            if (close[i] >= h3[i]) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wTrend_Filter_VolumeSpike"
timeframe = "12h"
leverage = 1.0