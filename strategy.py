#!/usr/bin/env python3
"""
6h Camarilla Pivot H3L3 Breakout + 12h EMA50 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels act as breakout points on 12h chart. Price breaking above H3 or below L3 with 12h EMA50 trend alignment and volume confirmation captures momentum moves. Works in bull/bear via trend filter and discrete sizing (0.25). Targets 50-150 trades over 4 years on 6h.
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
    
    # Load 12h data ONCE before loop for Camarilla pivots and EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h Camarilla pivot levels (based on previous 12h bar)
    # H4 = Close + 1.1*(High-Low)*1.5/2, L4 = Close - 1.1*(High-Low)*1.5/2
    # H3 = Close + 1.1*(High-Low)*1.25/2, L3 = Close - 1.1*(High-Low)*1.25/2
    # We'll use H3 and L3 as breakout levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot levels for each 12h bar
    camarilla_h3 = close_12h + 1.1 * (high_12h - low_12h) * 1.25 / 2
    camarilla_l3 = close_12h - 1.1 * (high_12h - low_12h) * 1.25 / 2
    
    # Align to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # 12h EMA50 for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 12h EMA50 warmup and volume MA
    start_idx = max(60, 21)  # EMA50 needs ~50, plus buffers
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 12h EMA50
        bullish_bias = curr_close > ema_12h_aligned[i]
        bearish_bias = curr_close < ema_12h_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H3/L3 breakout + trend + volume
            # Long: price breaks above camarilla H3 AND bullish bias AND volume spike
            long_entry = (curr_high > camarilla_h3_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below camarilla L3 AND bearish bias AND volume spike
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
            # Exit: price falls below camarilla L3 (invalidates breakout) OR loss of bullish bias
            if (curr_low < camarilla_l3_aligned[i]) or (curr_close < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above camarilla H3 (invalidates breakout) OR loss of bearish bias
            if (curr_high > camarilla_h3_aligned[i]) or (curr_close > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0