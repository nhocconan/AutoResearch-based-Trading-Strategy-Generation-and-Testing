#!/usr/bin/env python3
"""
1d_Donchian20_1wTrend_Filter_VolumeConfirmation
Hypothesis: Daily Donchian(20) breakout with weekly trend filter (price >/< weekly EMA34) and volume confirmation (>1.5x 20-day avg volume). 
Enters long on upper band breakout in weekly uptrend with volume spike, short on lower band breakout in weekly downtrend with volume spike. 
Exits on opposite band touch or trend reversal. Designed for 1d timeframe with ~10-25 trades/year, works in bull/bear by following weekly trend filter.
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
    
    # 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Donchian channels (20-period) on 1d timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need at least 20 bars for Donchian and EMA34 warmup
    start_idx = max(20, 34, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band in weekly uptrend with volume confirmation
            long_setup = (close[i] > highest_high[i]) and (close[i] > ema_34_1w_aligned[i]) and volume_spike[i]
            # Short: price breaks below lower Donchian band in weekly downtrend with volume confirmation
            short_setup = (close[i] < lowest_low[i]) and (close[i] < ema_34_1w_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches lower Donchian band OR weekly trend turns down
            if (close[i] <= lowest_low[i]) or (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches upper Donchian band OR weekly trend turns up
            if (close[i] >= highest_high[i]) or (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_1wTrend_Filter_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0