#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h primary with 12h HTF - Donchian(20) breakout with volume confirmation and trend filter
    # Designed to capture medium-term breakouts with institutional volume confirmation
    # Target: 60-120 trades over 4 years (15-30/year) for low fee drag
    # Works in both bull and bear markets by trading breakouts in direction of 12h EMA50 trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 12h data for HTF Donchian channels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 6h data for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values if 'volume' in df_6h.columns else np.ones(len(df_6h))
    
    # Calculate 12h Donchian channels (20-period)
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_12h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_20)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_6h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume_6h[i] > 1.5 * vol_avg_20_aligned[i]
        
        # Breakout conditions at Donchian channels
        breakout_up = close[i] > highest_20_aligned[i]
        breakout_down = close[i] < lowest_20_aligned[i]
        
        # Trend filter: only trade in direction of 12h EMA50
        trend_filter_long = close[i] > ema50_12h_aligned[i]
        trend_filter_short = close[i] < ema50_12h_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and volume_confirmed and trend_filter_long
        enter_short = breakout_down and volume_confirmed and trend_filter_short
        
        # Exit conditions: opposite Donchian breakout or trend reversal
        exit_long = position == 1 and (close[i] < lowest_20_aligned[i] or close[i] < ema50_12h_aligned[i])
        exit_short = position == -1 and (close[i] > highest_20_aligned[i] or close[i] > ema50_12h_aligned[i])
        
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

name = "6h_12h_donchian_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0