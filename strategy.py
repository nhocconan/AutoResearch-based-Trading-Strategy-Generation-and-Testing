#!/usr/bin/env python3
"""
12h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
Hypothesis: 12h Donchian breakouts capture medium-term momentum. 1d EMA50 filter ensures alignment with daily trend. Volume confirmation reduces false breakouts. Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years). Works in both bull and bear markets by following 1d trend and avoiding counter-trend entries.
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
    
    # Load 1d data ONCE before loop for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from 1d
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 1.8 * 20-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 50, 20)  # Donchian needs 20 bars, EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA50
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: breakout + trend + volume
            # Long: price breaks above Donchian high AND bullish bias AND volume spike
            long_entry = (curr_high > donchian_high_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below Donchian low AND bearish bias AND volume spike
            short_entry = (curr_low < donchian_low_aligned[i]) and bearish_bias and vol_spike
            
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
            # Exit: price falls below Donchian low (mean reversion) OR loss of bullish bias
            if (curr_low < donchian_low_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian high (mean reversion) OR loss of bearish bias
            if (curr_high > donchian_high_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0