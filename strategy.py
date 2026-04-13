#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
    # Designed to capture strong directional moves filtered by higher timeframe trend
    # Works in bull markets via upside breakouts and bear markets via downside breakouts
    # Target: 100-150 trades over 4 years (25-38/year) for low fee drag and good generalization
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values if 'volume' in df_4h.columns else np.ones(len(df_4h))
    
    # Calculate 1d EMA50 for trend filter (responsive trend)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h primary timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume_4h[i] > 1.5 * vol_avg_20_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > highest_high_20[i-1]  # Break above 20-period high
        breakout_short = close[i] < lowest_low_20[i-1]   # Break below 20-period low
        
        # Trend filter: only trade in direction of 1d EMA50
        trend_filter_long = close[i] > ema50_1d_aligned[i]
        trend_filter_short = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions
        enter_long = breakout_long and volume_confirmed and trend_filter_long
        enter_short = breakout_short and volume_confirmed and trend_filter_short
        
        # Exit conditions: opposite Donchian breakout or trend reversal
        exit_long = position == 1 and (close[i] < lowest_low_20[i] or close[i] < ema50_1d_aligned[i])
        exit_short = position == -1 and (close[i] > highest_high_20[i] or close[i] > ema50_1d_aligned[i])
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
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

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0