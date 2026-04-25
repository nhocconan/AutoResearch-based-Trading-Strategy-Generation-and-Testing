#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + Weekly EMA34 Trend + Volume Spike
Hypothesis: Daily Donchian breakouts capture medium-term trends, filtered by weekly EMA trend direction and volume confirmation. Works in bull/bear via trend filter. Target: 10-25 trades/year on 1d.
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
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Weekly EMA34 for trend direction
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and weekly EMA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price above/below weekly EMA34
        bullish_trend = curr_close > ema_34_1w_aligned[i]
        bearish_trend = curr_close < ema_34_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + trend + volume
            # Long: price breaks above upper Donchian AND bullish trend AND volume spike
            long_entry = (curr_high > high_roll[i]) and bullish_trend and vol_spike
            # Short: price breaks below lower Donchian AND bearish trend AND volume spike
            short_entry = (curr_low < low_roll[i]) and bearish_trend and vol_spike
            
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
            # Exit: price breaks below lower Donchian OR trend changes
            if (curr_low < low_roll[i]) or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above upper Donchian OR trend changes
            if (curr_high > high_roll[i]) or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0