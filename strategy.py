#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout + 12h EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels from 12h act as significant support/resistance. 
Break above H3 with volume and 12h EMA34 uptrend signals bullish momentum.
Break below L3 with volume and 12h EMA34 downtrend signals bearish momentum.
Uses 4h timeframe for balanced trade frequency. Works in bull/bear via EMA trend filter.
Volume spike confirms institutional participation. Target: 20-50 trades/year.
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
    
    # Get 12h data for Camarilla pivot calculation and EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need sufficient data for pivot and EMA
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (based on previous bar's OHLC)
    # Camarilla: H3 = C + ((H-L)*1.1/4), L3 = C - ((H-L)*1.1/4)
    # We use the previous bar's OHLC to avoid look-ahead
    prev_close = df_12h['close'].shift(1).values
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    
    # Calculate pivot levels using previous bar's data
    range_hl = prev_high - prev_low
    camarilla_h3 = prev_close + (range_hl * 1.1 / 4)
    camarilla_l3 = prev_close - (range_hl * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed as they're based on prev bar)
    h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # Calculate 12h EMA34 for trend filter
    if len(df_12h) >= 34:
        ema_34 = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
        ema34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    else:
        ema34_aligned = np.full(n, close[0])  # default to first price if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        ema34_value = ema34_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Trend filter: price above/below EMA34
        uptrend = curr_close > ema34_value
        downtrend = curr_close < ema34_value
        
        if position == 0:
            # Long: price breaks above H3 AND volume spike AND uptrend
            long_condition = (curr_close > h3_level) and volume_spike and uptrend
            # Short: price breaks below L3 AND volume spike AND downtrend
            short_condition = (curr_close < l3_level) and volume_spike and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below L3 or trend turns down
            if curr_close <= l3_level or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above H3 or trend turns up
            if curr_close >= h3_level or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0