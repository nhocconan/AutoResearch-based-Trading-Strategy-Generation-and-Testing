#!/usr/bin/env python3
"""
4h Donchian(20) Breakout with Volume Spike and ADX Trend Filter
Hypothesis: Donchian channel breakouts capture strong momentum moves. 
Volume confirmation ensures participation, while ADX > 25 filters for trending markets.
This combination works in both bull and bear markets by catching breakouts in the direction 
of the trend. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 20-50 trades/year on 4h.
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
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ADX calculation for trend filter (14-period)
    plus_dm = pd.Series(high).diff()
    minus_dm = pd.Series(low).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = pd.Series(high).sub(pd.Series(low))
    tr2 = pd.Series(high).sub(pd.Series(close).shift(1)).abs()
    tr3 = pd.Series(low).sub(pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(lookback, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        trend_filter = adx[i] > 25  # ADX > 25 indicates trending market
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian AND volume spike AND trending market
            long_entry = (curr_close > upper[i]) and vol_spike and trend_filter
            # Short: price breaks below lower Donchian AND volume spike AND trending market
            short_entry = (curr_close < lower[i]) and vol_spike and trend_filter
            
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
            # Exit: price falls below lower Donchian (breakdown) OR loss of momentum (ADX < 20)
            if (curr_close < lower[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above upper Donchian (breakout) OR loss of momentum (ADX < 20)
            if (curr_close > upper[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ADXTrend"
timeframe = "4h"
leverage = 1.0