#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dTrend_VolumeConfirm
Hypothesis: 12h Camarilla H3/L3 breakout with 1d trend filter (price >/< EMA34) and volume confirmation (>2.0x 20-bar avg). 
Enters long when price breaks above H3 level in 1d uptrend with volume spike, short when price breaks below L3 level in 1d downtrend with volume spike. 
Exits on opposite Camarilla level touch (L3 for longs exit, H3 for shorts exit) or trend reversal. 
Designed for 12h timeframe with ~12-30 trades/year, works in bull/bear by following 1d trend filter and momentum confirmation.
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
    open_price = prices['open'].values
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's Camarilla levels (using 1d OHLC)
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_range = (high_1d - low_1d) * 1.1 / 2.0
    h3_level = close_1d + camarilla_range
    l3_level = close_1d - camarilla_range
    
    # Align HTF Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_level)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_level)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need at least 1 bar of previous data and EMA34 warmup
    start_idx = max(34, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 level in 1d uptrend with volume confirmation
            long_setup = (open_price[i] <= h3_aligned[i] and close[i] > h3_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and volume_spike[i]
            # Short: price breaks below L3 level in 1d downtrend with volume confirmation
            short_setup = (open_price[i] >= l3_aligned[i] and close[i] < l3_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches L3 level (mean reversion) OR trend turns down
            if (close[i] < l3_aligned[i]) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches H3 level (mean reversion) OR trend turns up
            if (close[i] > h3_aligned[i]) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0