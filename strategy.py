#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
    # Designed for very low trade frequency (5-15/year) to minimize fee drag on 1d timeframe
    # Uses 1w for trend direction, 1d for entry/execution, volume spike for confirmation
    # Works in bull (breakouts with trend) and bear (avoids counter-trend breakouts via 1w filter)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Donchian channels (20-period)
    # Upper band: highest high over past 20 days
    # Lower band: lowest low over past 20 days
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 1d primary timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * vol_avg_20_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > high_20_aligned[i]  # Price above upper Donchian -> long
        breakout_short = close[i] < low_20_aligned[i]  # Price below lower Donchian -> short
        
        # Trend filter: only trade in direction of 1w EMA50
        # For long: price above EMA50; for short: price below EMA50
        trend_filter_long = close[i] > ema50_1w_aligned[i]
        trend_filter_short = close[i] < ema50_1w_aligned[i]
        
        # Entry conditions
        enter_long = breakout_long and volume_confirmed and trend_filter_long
        enter_short = breakout_short and volume_confirmed and trend_filter_short
        
        # Exit conditions: opposite Donchian breakout or volume dry-up
        exit_long = position == 1 and (close[i] < low_20_aligned[i] or volume[i] < 0.5 * vol_avg_20_aligned[i])
        exit_short = position == -1 and (close[i] > high_20_aligned[i] or volume[i] < 0.5 * vol_avg_20_aligned[i])
        
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