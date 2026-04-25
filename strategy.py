#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels on 12h act as strong support/resistance; breakouts with volume confirmation and weekly EMA50 trend alignment capture momentum moves in both bull and bear markets. Designed for 12h to target 12-37 trades/year (50-150 over 4 years), minimizing fee drag. Uses 1w trend filter to avoid counter-trend entries and reduce whipsaw.
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
    
    # Load 12h data ONCE before loop for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (H3, L3) from 12h
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    hl_range = df_12h['high'] - df_12h['low']
    camarilla_h3 = df_12h['close'] + (1.1 * hl_range / 4)
    camarilla_l3 = df_12h['close'] - (1.1 * hl_range / 4)
    
    # Align Camarilla levels to primary timeframe (assumed 12h, but works for lower TF too)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3.values)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 50, 20)  # volume MA, EMA50, Camarilla needs 1 bar
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1w EMA50
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: breakout + trend + volume
            # Long: price breaks above Camarilla H3 AND bullish bias AND volume spike
            long_entry = (curr_high > camarilla_h3_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below Camarilla L3 AND bearish bias AND volume spike
            short_entry = (curr_low < camarilla_l3_aligned[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Camarilla L3 (mean reversion) OR loss of bullish bias
            if (curr_low < camarilla_l3_aligned[i]) or (curr_close < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Camarilla H3 (mean reversion) OR loss of bearish bias
            if (curr_high > camarilla_h3_aligned[i]) or (curr_close > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0