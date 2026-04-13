#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w EMA50 trend filter
    # Designed to capture weekly trend continuation breakouts at key daily levels with volume confirmation
    # Target: 30-80 trades over 4 years (7-20/year) for low fee drag and good generalization
    # Works in both bull and bear markets by trading breakouts in direction of 1w trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Donchian levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for volume confirmation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values if 'volume' in df_1w.columns else np.ones(len(df_1w))
    
    # Calculate 1d Donchian(20) channels (based on previous 20 days)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w volume average (10-period)
    vol_avg_10 = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    
    # Align all HTF indicators to 1d primary timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_avg_10_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_avg_10_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 10-period 1w average
        volume_confirmed = volume[i] > 1.5 * vol_avg_10_aligned[i]
        
        # Breakout conditions at 1d Donchian channels
        breakout_up = close[i] > high_20_aligned[i]
        breakout_down = close[i] < low_20_aligned[i]
        
        # Trend filter: only trade in direction of 1w EMA50
        # For long: price above EMA50; for short: price below EMA50
        trend_filter_long = close[i] > ema50_1w_aligned[i]
        trend_filter_short = close[i] < ema50_1w_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and volume_confirmed and trend_filter_long
        enter_short = breakout_down and volume_confirmed and trend_filter_short
        
        # Exit conditions: price returns to opposite Donchian channel (mean reversion)
        exit_long = position == 1 and close[i] < low_20_aligned[i]
        exit_short = position == -1 and close[i] > high_20_aligned[i]
        
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