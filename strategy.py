#!/usr/bin/env python3
"""
6h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance on 6h charts.
Breakout above H3 with volume spike and 1d EMA34 uptrend signals bullish momentum.
Breakdown below L3 with volume spike and 1d EMA34 downtrend signals bearish momentum.
Volume spike confirms institutional participation. Works in trending markets.
6h timeframe targets 12-37 trades/year (50-150 over 4 years) with tight entry conditions.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for 6h timeframe using daily OHLC
    # H3 = Close + 1.1 * (High - Low) / 2
    # L3 = Close - 1.1 * (High - Low) / 2
    camarilla_h3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_l3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34)  # volume MA, 1d EMA34 alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: break above H3 with volume spike AND uptrend
            long_entry = (curr_high > h3_aligned[i]) and vol_spike and uptrend
            # Short: break below L3 with volume spike AND downtrend
            short_entry = (curr_low < l3_aligned[i]) and vol_spike and downtrend
            
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
            # Exit: price breaks below L3 (failed bullish structure) OR loss of uptrend
            if (curr_low < l3_aligned[i]) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above H3 (failed bearish structure) OR loss of downtrend
            if (curr_high > h3_aligned[i]) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0