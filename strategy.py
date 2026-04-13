#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w trend filter
    # Designed for very low trade frequency (12-37/year) to minimize fee drag on 12h timeframe
    # Uses 1d for volume confirmation, 1w for trend direction, 12h only for entry timing
    # Works in both bull and bear: breakouts in trend direction, volume confirms conviction
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(vol_avg_20_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        # Get the 1d bar index for current 12h bar (each 1d bar = 2 12h bars)
        idx_1d = i // 2
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 2.0 * vol_avg_20_1d_aligned[i]
        
        # Donchian breakout conditions (20-period)
        lookback = min(i, 20)
        if lookback < 20:
            signals[i] = 0.0 if position == 0 else (position_size if position == 1 else -position_size)
            continue
            
        highest_high = np.max(high[i-lookback:i])
        lowest_low = np.min(low[i-lookback:i])
        
        # Breakout conditions
        breakout_long = close[i] > highest_high and volume_confirmed
        breakout_short = close[i] < lowest_low and volume_confirmed
        
        # Trend filter: only trade in direction of 1w EMA50
        trend_filter_long = close[i] > ema50_1w_aligned[i]
        trend_filter_short = close[i] < ema50_1w_aligned[i]
        
        # Entry conditions
        enter_long = breakout_long and trend_filter_long
        enter_short = breakout_short and trend_filter_short
        
        # Exit conditions: opposite Donchian breakout or trend reversal
        exit_long = position == 1 and (close[i] < lowest_low or close[i] < ema50_1w_aligned[i])
        exit_short = position == -1 and (close[i] > highest_high or close[i] > ema50_1w_aligned[i])
        
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

name = "12h_1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0