#!/usr/bin/env python3
"""
6h Camarilla R3/S3 Breakout + 12h EMA50 Trend + Volume Spike
Hypothesis: Camarilla R3/S3 levels on 12h identify key intraday support/resistance; breakouts with 12h EMA50 trend filter and volume confirmation capture momentum swings. Designed for 6h timeframe to target 12-37 trades/year (50-150 over 4 years), minimizing fee drag. Works in both bull and bear markets by following the 12h trend and avoiding counter-trend entries.
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 12h
    # Typical price = (H+L+C)/3
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    # Camarilla width = (H - L) * 1.1 / 12
    camarilla_width = (df_12h['high'] - df_12h['low']) * 1.1 / 12
    # R3 = C + width * 1.1, S3 = C - width * 1.1
    r3 = typical_price + camarilla_width * 1.1
    s3 = typical_price - camarilla_width * 1.1
    # R4 = C + width * 1.5, S4 = C - width * 1.5 (stronger breakout levels)
    r4 = typical_price + camarilla_width * 1.5
    s4 = typical_price - camarilla_width * 1.5
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3.values)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4.values)
    
    # 12h EMA50 for trend filter
    ema_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter for 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 50)  # volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_12h_aligned[i])):
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
            # Look for entry signals
            # Long: price breaks above R3/R4 AND bullish bias AND volume spike
            long_entry = ((curr_high > r3_aligned[i]) or (curr_high > r4_aligned[i])) and bullish_bias and vol_spike
            # Short: price breaks below S3/S4 AND bearish bias AND volume spike
            short_entry = ((curr_low < s3_aligned[i]) or (curr_low < s4_aligned[i])) and bearish_bias and vol_spike
            
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
            # Exit: price falls below S3 (mean reversion) OR loss of bullish bias
            if (curr_low < s3_aligned[i]) or (curr_close < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above R3 (mean reversion) OR loss of bearish bias
            if (curr_high > r3_aligned[i]) or (curr_close > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0