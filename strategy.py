#!/usr/bin/env python3
"""
1d_Camarilla_H3L3_Breakout_1wTrend_VolumeSpike
Hypothesis: On daily timeframe, Camarilla H3/L3 breakouts capture strong momentum with weekly trend confirmation.
Break above H3 with volume spike and weekly uptrend (price > weekly EMA34) signals long;
break below L3 with volume spike and weekly downtrend (price < weekly EMA34) signals short.
Uses fixed position size (0.25) to limit trades (~15-25/year) and minimize fee drag.
Designed for BTC/ETH to work in both bull and bear markets by trading breakouts with trend and volume confirmation.
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
    
    # 1d data for Camarilla calculation (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # H3 = close + 1.1*(high - low)/2
    # L3 = close - 1.1*(high - low)/2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 1d timeframe (they are already 1d)
    camarilla_h3_aligned = camarilla_h3  # already aligned to 1d bars
    camarilla_l3_aligned = camarilla_l3  # already aligned to 1d bars
    
    # 1w data for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA34 for trend filter (loaded ONCE)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1d volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need Camarilla (requires 1d data, so 1 bar), volume MA (20), weekly EMA (34)
    start_idx = max(1, 20, 34)  # at least 34 for weekly EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above Camarilla H3 with volume spike and weekly uptrend
            long_breakout = (curr_close > camarilla_h3_aligned[i]) and vol_spike[i] and (curr_close > ema_34_1w_aligned[i])
            # Short: price breaks below Camarilla L3 with volume spike and weekly downtrend
            short_breakout = (curr_close < camarilla_l3_aligned[i]) and vol_spike[i] and (curr_close < ema_34_1w_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below Camarilla L3 OR weekly trend turns down
            if (curr_close < camarilla_l3_aligned[i]) or (curr_close < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Camarilla H3 OR weekly trend turns up
            if (curr_close > camarilla_h3_aligned[i]) or (curr_close > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0