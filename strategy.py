#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
    # Long: price breaks above Donchian(20) high + price > 1w EMA50 (uptrend) + volume > 1.5x average
    # Short: price breaks below Donchian(20) low + price < 1w EMA50 (downtrend) + volume > 1.5x average
    # Exit: Donchian(10) opposite breakout or volume drops below average
    # Uses 1w EMA for trend alignment to avoid counter-trend trades
    # Donchian captures breakouts, volume confirms strength, 1w EMA filters trend direction
    # Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
    # Target: 40-80 total trades over 4 years (10-20/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data (primary timeframe)
    df_1d = prices  # prices is already 1d timeframe
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume confirmation (1.5x 20-period average)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_ma_20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for Donchian
        # Skip if data not ready
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_high_20[i]
        short_breakout = close[i] < lowest_low_20[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > volume_threshold[i]
        
        # Trend filter from 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and uptrend and volume_confirmed and position != 1
        short_entry = short_breakout and downtrend and volume_confirmed and position != -1
        
        # Exit conditions: Donchian(10) opposite breakout or volume drops below average
        exit_long = position == 1 and (close[i] < lowest_low_10[i] or volume[i] < volume_ma_20[i])
        exit_short = position == -1 and (close[i] > highest_high_10[i] or volume[i] < volume_ma_20[i])
        
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

name = "1d_1w_donchian_breakout_trend_volume_v1"
timeframe = "1d"
leverage = 1.0