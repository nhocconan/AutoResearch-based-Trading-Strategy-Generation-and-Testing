#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian channel breakout with 1w EMA trend filter and volume confirmation
    # Long: price breaks above 20-period Donchian upper + price > 1w EMA50 + volume > 1.5x avg
    # Short: price breaks below 20-period Donchian lower + price < 1w EMA50 + volume > 1.5x avg
    # Exit: price returns to 20-period Donchian middle (midpoint of upper/lower)
    # Uses 12h primary timeframe for lower trade frequency, suitable for 12h constraints
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    # Donchian breakouts capture strong moves; EMA filter ensures trend alignment; volume confirms validity
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 12h data for primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.ones(len(df_12h))
    
    # Get 1w data for EMA trend filter (MTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Donchian Channel on 12h data (20-period)
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate EMA50 on 1w data for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average on 12h data
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # start from 50 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_avg_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close_12h[i] > donchian_upper_aligned[i]
        breakout_down = close_12h[i] < donchian_lower_aligned[i]
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = close_12h[i] > ema_50_1w_aligned[i]
        price_below_ema = close_12h[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume_12h[i] > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and price_above_ema and volume_confirmed
        enter_short = breakout_down and price_below_ema and volume_confirmed
        
        # Exit conditions: price returns to Donchian middle
        exit_long = position == 1 and close_12h[i] <= donchian_middle_aligned[i]
        exit_short = position == -1 and close_12h[i] >= donchian_middle_aligned[i]
        
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

name = "12h_1w_donchian_ema_breakout_v1"
timeframe = "12h"
leverage = 1.0