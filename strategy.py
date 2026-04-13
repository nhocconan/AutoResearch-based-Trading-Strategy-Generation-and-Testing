#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R + 12h EMA trend filter + volume confirmation
    # Long when: Williams %R(14) crosses above -80 (oversold bounce) AND price > 12h EMA(50) (uptrend) AND volume > 1.3x 20-bar avg
    # Short when: Williams %R(14) crosses below -20 (overbought rejection) AND price < 12h EMA(50) (downtrend) AND volume > 1.3x 20-bar avg
    # Exit when: Williams %R crosses above -20 (long exit) or below -80 (short exit)
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years (12-37/year).
    # Williams %R identifies exhaustion points; 12h EMA ensures we trade with the higher timeframe trend.
    # Volume confirmation reduces false signals. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Williams %R for 6h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 6h Williams %R to 15m timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 12h data for EMA trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) for 12h
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume confirmation: volume > 1.3x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.3 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R crossover conditions (using current vs previous bar)
        wr_cross_up = williams_r_aligned[i-1] <= -80 and williams_r_aligned[i] > -80  # cross above -80
        wr_cross_down = williams_r_aligned[i-1] >= -20 and williams_r_aligned[i] < -20  # cross below -20
        
        # Trend filter: price vs 12h EMA(50)
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        # Entry conditions with trend filter and volume confirmation
        long_entry = wr_cross_up and price_above_ema and volume_confirmed[i] and position != 1
        short_entry = wr_cross_down and price_below_ema and volume_confirmed[i] and position != -1
        
        # Exit conditions: Williams %R crosses opposite threshold
        exit_long = (position == 1 and williams_r_aligned[i] > -20)
        exit_short = (position == -1 and williams_r_aligned[i] < -80)
        
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

name = "6h_12h_williams_r_ema_volume_v1"
timeframe = "6h"
leverage = 1.0