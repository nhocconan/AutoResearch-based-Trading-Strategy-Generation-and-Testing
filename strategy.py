#!/usr/bin/env python3
"""
4h Donchian Breakout + 12h EMA Trend + Volume Spike
Hypothesis: Donchian(20) breakouts capture institutional moves. 12h EMA50 trend filter ensures alignment with higher timeframe momentum. Volume confirmation avoids false breakouts. Works in bull/bear via discrete sizing (0.25) and trend filter. Fewer trades (<50/year) to minimize fee drag.
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
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian(20) channels from primary timeframe
    def calculate_donchian(high_arr, low_arr, window=20):
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 12h EMA warmup and Donchian/volume MA
    start_idx = max(70, 50)  # EMA50 needs ~50, Donchian 20, but using 70 for safety
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
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
            # Look for entry signals - require: Donchian breakout + trend + volume
            # Long: price breaks above Donchian upper AND bullish bias AND volume spike
            long_entry = (curr_high > donchian_upper[i]) and bullish_bias and vol_spike
            # Short: price breaks below Donchian lower AND bearish bias AND volume spike
            short_entry = (curr_low < donchian_lower[i]) and bearish_bias and vol_spike
            
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
            # Exit: price falls below Donchian lower (mean reversion) OR loss of bullish bias
            if (curr_low < donchian_lower[i]) or (curr_close < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian upper (mean reversion) OR loss of bearish bias
            if (curr_high > donchian_upper[i]) or (curr_close > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0