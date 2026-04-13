#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w EMA50 trend filter
    # Designed to capture strong weekly trend continuation breakouts with institutional volume
    # Target: 40-80 trades over 4 years (10-20/year) for low fee drag and good generalization
    # Works in both bull and bear markets by trading breakouts in direction of 1w trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Donchian levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian(20) channels (based on previous 20 days)
    # Using rolling window with min_periods=20, shift by 1 to avoid look-ahead
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1d 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 1d primary timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_roll)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_roll)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume_1d[i] > 1.8 * vol_avg_20_aligned[i]
        
        # Breakout conditions at Donchian channels
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Trend filter: only trade in direction of 1w EMA50
        trend_filter_long = close[i] > ema50_1w_aligned[i]
        trend_filter_short = close[i] < ema50_1w_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and volume_confirmed and trend_filter_long
        enter_short = breakout_down and volume_confirmed and trend_filter_short
        
        # Exit conditions: opposite Donchian breakout or price returns to midpoint
        midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
        exit_long = position == 1 and (close[i] < midpoint or breakout_down)
        exit_short = position == -1 and (close[i] > midpoint or breakout_up)
        
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

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0