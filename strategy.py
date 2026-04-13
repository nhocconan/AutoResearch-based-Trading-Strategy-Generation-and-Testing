#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w trend filter
    # Designed for low trade frequency (19-50/year) to minimize fee drag
    # Works in bull/bear markets by capturing breakouts with volume confirmation and higher timeframe trend alignment
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for volume confirmation and 1w data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d volume average (20-period) for spike detection
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align all HTF indicators to 4h primary timeframe
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(vol_avg_20_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average (tight filter)
        volume_confirmed = volume_1d[i] > 2.0 * vol_avg_20_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > donchian_high[i-1]  # Break above previous period high
        breakout_short = close[i] < donchian_low[i-1]  # Break below previous period low
        
        # Trend filter: only trade in direction of 1w EMA50
        trend_filter_long = close[i] > ema50_1w_aligned[i]
        trend_filter_short = close[i] < ema50_1w_aligned[i]
        
        # Entry conditions (tight: volume + breakout + trend)
        enter_long = breakout_long and volume_confirmed and trend_filter_long
        enter_short = breakout_short and volume_confirmed and trend_filter_short
        
        # Exit conditions: opposite Donchian breakout or volume dry-up
        exit_long = position == 1 and (close[i] < donchian_low[i] or volume_1d[i] < 0.5 * vol_avg_20_1d_aligned[i])
        exit_short = position == -1 and (close[i] > donchian_high[i] or volume_1d[i] < 0.5 * vol_avg_20_1d_aligned[i])
        
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

name = "4h_1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0