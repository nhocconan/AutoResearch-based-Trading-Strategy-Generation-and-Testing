#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_12hTrend_VolumeConfirm
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume confirmation (>1.5x 20-bar avg). 
Enters long when price breaks above H3 with 12h uptrend (price > EMA34) and volume spike, short when price breaks below L3 with 12h downtrend (price < EMA34) and volume spike. 
Exits on opposite Camarilla level touch (L3 for longs exit, H3 for shorts exit) or trend reversal. 
Designed for 4h timeframe with ~20-50 trades/year, works in bull/bear by following 12h trend filter.
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
    
    # 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Previous day's high, low, close for Camarilla levels (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need at least 1 bar of previous data and warmup for indicators
    start_idx = max(34, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 with 12h uptrend and volume confirmation
            long_setup = (close[i] > camarilla_h3_aligned[i]) and (close[i-1] <= camarilla_h3_aligned[i-1]) and \
                         (close[i] > ema_34_12h_aligned[i]) and volume_spike[i]
            # Short: price breaks below L3 with 12h downtrend and volume confirmation
            short_setup = (close[i] < camarilla_l3_aligned[i]) and (close[i-1] >= camarilla_l3_aligned[i-1]) and \
                          (close[i] < ema_34_12h_aligned[i]) and volume_spike[i]
            
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
            # Exit: price touches L3 (opposite level) OR trend turns down
            if (close[i] < camarilla_l3_aligned[i]) or (close[i] < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches H3 (opposite level) OR trend turns up
            if (close[i] > camarilla_h3_aligned[i]) or (close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0