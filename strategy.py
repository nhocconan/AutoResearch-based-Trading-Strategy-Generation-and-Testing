#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1w EMA34 Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum. 1w EMA34 ensures we trade with the primary trend (works in bull/bear by only taking trend-aligned breakouts). Volume confirmation filters weak breakouts. Discrete sizing (0.25) minimizes fee churn. Targets 50-150 trades over 4 years.
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
    
    # Get 1w data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period Donchian channels on 12h
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-19:i+1])
        lowest_20[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for 12h volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, EMA, and volume MA
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current 12h volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: Break above upper Donchian AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_high > upper_channel and 
                         curr_close > ema_trend and volume_confirm)
            # Short: Break below lower Donchian AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_low < lower_channel and 
                          curr_close < ema_trend and volume_confirm)
            
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
            # Exit: Price crosses below lower Donchian OR EMA34 trend turns down
            if (curr_close < lower_channel or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Price crosses above upper Donchian OR EMA34 trend turns up
            if (curr_close > upper_channel or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0