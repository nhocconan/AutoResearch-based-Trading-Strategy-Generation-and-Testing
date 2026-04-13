#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation
    # Long: price breaks above Donchian(20) high AND price > 1w EMA200 AND volume > 1.5x 20-period average
    # Short: price breaks below Donchian(20) low AND price < 1w EMA200 AND volume > 1.5x 20-period average
    # Exit: opposite Donchian breakout OR EMA200 trend reversal
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Works in bull/bear via 1w EMA200 trend filter preventing counter-trend trades.
    # Volume confirmation reduces false breakouts.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA200
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate Donchian channels (20-period) on 12h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < lowest_low[i-1]  # Break below previous period's low
        
        # 1w EMA200 trend filter
        uptrend = close[i] > ema200_1w_aligned[i]
        downtrend = close[i] < ema200_1w_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and uptrend and volume_confirmed[i] and position != 1
        short_entry = breakout_down and downtrend and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (close[i] < lowest_low[i-1]) or (position == 1 and not uptrend)
        exit_short = (close[i] > highest_high[i-1]) or (position == -1 and not downtrend)
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0