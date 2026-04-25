#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with Volume Spike and Weekly EMA34 Trend Filter
Hypothesis: Donchian channel breakouts on daily timeframe capture major momentum moves.
Volume confirmation ensures participation, while weekly EMA34 filters for higher timeframe trend.
This combination works in both bull and bear markets by catching breakouts in the direction 
of the weekly trend. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 30-100 trades over 4 years.
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
    
    # Calculate Donchian channels (20-period) - using prior close to avoid look-ahead
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    # Calculate EMA34 on weekly closes
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align to lower timeframe (1d) - already waits for weekly bar close
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(lookback, 20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        weekly_trend_up = close[i] > ema_34_1w_aligned[i]  # Price above weekly EMA34
        weekly_trend_down = close[i] < ema_34_1w_aligned[i]  # Price below weekly EMA34
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian AND volume spike AND weekly uptrend
            long_entry = (curr_close > upper[i]) and vol_spike and weekly_trend_up
            # Short: price breaks below lower Donchian AND volume spike AND weekly downtrend
            short_entry = (curr_close < lower[i]) and vol_spike and weekly_trend_down
            
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
            # Exit: price falls below lower Donchian (breakdown) OR weekly trend turns down
            if (curr_close < lower[i]) or (not weekly_trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above upper Donchian (breakout) OR weekly trend turns up
            if (curr_close > upper[i]) or (not weekly_trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_VolumeSpike_WeeklyEMA34_Trend"
timeframe = "1d"
leverage = 1.0