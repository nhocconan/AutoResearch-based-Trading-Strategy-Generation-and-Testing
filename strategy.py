#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d HTF for EMA34 to capture medium-term trend and reduce false breakouts.
# Donchian(20) from 12h provides proven price channel structure for breakouts.
# Volume confirmation at 1.8x average ensures strong participation while limiting trades (~12-30/year target).
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
# Discrete sizing 0.25 to balance opportunity and fee drag. Works in bull/bear: trend filter ensures trades only with momentum.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag and improve test generalization.

name = "12h_Donchian20_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Donchian(20) levels from 12h timeframe (using prior completed 12h bar)
    # For 12h timeframe, we use shift(1) on the same timeframe to get prior completed bar
    if len(prices) < 2:
        return np.zeros(n)
    
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    
    # Donchian(20) upper and lower bands from prior 20 completed 12h bars
    lookback = 20
    highest_high = pd.Series(prev_high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(prev_low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1d EMA34 for trend filter (medium-term trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 1.8x 20-period average (strict threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper AND price > 1d EMA34 AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower AND price < 1d EMA34 AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below Donchian lower OR price < 1d EMA34
            if close[i] < lowest_low[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above Donchian upper OR price > 1d EMA34
            if close[i] > highest_high[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals