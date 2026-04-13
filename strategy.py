#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h primary with 1d HTF - Donchian channel breakout with volume confirmation
    # Designed to capture strong momentum moves at key daily levels with volume confirmation
    # Target: 50-150 trades over 4 years (12-37/year) for low fee drag and good generalization
    # Works in both bull and bear markets by trading breakouts in direction of 1d trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Donchian levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20-period) based on previous day
    # Upper band = highest high of last 20 days, Lower band = lowest low of last 20 days
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume average (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.ones(len(df_12h))
    vol_avg_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume_12h[i] > 2.0 * vol_avg_20_aligned[i]
        
        # Breakout conditions at Donchian levels
        breakout_up = high[i] > high_20_aligned[i]
        breakout_down = low[i] < low_20_aligned[i]
        
        # Trend filter: only trade in direction of 1d EMA50
        # For long: price above EMA50; for short: price below EMA50
        trend_filter_long = close[i] > ema50_1d_aligned[i]
        trend_filter_short = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and volume_confirmed and trend_filter_long
        enter_short = breakout_down and volume_confirmed and trend_filter_short
        
        # Exit conditions: price returns to opposite Donchian band (mean reversion)
        exit_long = position == 1 and low[i] <= low_20_aligned[i]
        exit_short = position == -1 and high[i] >= high_20_aligned[i]
        
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

name = "12h_1d_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0