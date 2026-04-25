#!/usr/bin/env python3
"""
6h Camarilla Pivot H3/L3 Breakout + 1d EMA50 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels from daily timeframe represent stronger support/resistance than R1/S1.
Breakouts above H3 or below L3 with volume confirmation and aligned with 1d EMA50 trend capture significant momentum moves.
Designed for 6h timeframe to achieve 12-37 trades/year (50-150 total over 4 years) with tight entry conditions.
Works in bull (breakouts above H3 in uptrend) and bear (breakouts below L3 in downtrend).
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (High + Low + Close) / 3
    # H3 = Close + (High - Low) * 1.1 / 4
    # L3 = Close - (High - Low) * 1.1 / 4
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    h3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    l3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 6h timeframe (no extra delay needed for pivot levels)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Calculate EMA50 on 1d close for trend
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        h3_level = h3_1d_aligned[i]
        l3_level = l3_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND price > EMA (uptrend)
            long_entry = (curr_high > h3_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 AND volume spike AND price < EMA (downtrend)
            short_entry = (curr_low < l3_level) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price crosses below H3 OR price crosses below EMA (trend change)
            if (curr_low < h3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above L3 OR price crosses above EMA (trend change)
            if (curr_high > l3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0