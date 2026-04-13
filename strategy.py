#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme + 1d EMA200 trend filter + volume spike
    # Williams %R identifies overbought/oversold conditions (below -80 = oversold, above -20 = overbought)
    # In strong trends (price > EMA200 for longs, < EMA200 for shorts), extreme %R often precedes continuation
    # Volume spike confirms institutional interest at extreme levels
    # Designed for low frequency: ~25-40 trades/year to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Williams %R, EMA200, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d volume average (20-period) for spike detection
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average (significant spike)
        volume_spike = volume_1d[i] > 2.0 * vol_avg_20_1d_aligned[i]
        
        # Williams %R extreme conditions
        wr_oversold = williams_r_aligned[i] < -80   # Oversold - potential long
        wr_overbought = williams_r_aligned[i] > -20 # Overbought - potential short
        
        # Trend filter: only trade in direction of 1d EMA200
        trend_filter_long = close[i] > ema200_1d_aligned[i]   # Uptrend
        trend_filter_short = close[i] < ema200_1d_aligned[i]  # Downtrend
        
        # Entry conditions: extreme %R + volume spike + trend alignment
        enter_long = wr_oversold and volume_spike and trend_filter_long
        enter_short = wr_overbought and volume_spike and trend_filter_short
        
        # Exit conditions: %R returns to neutral territory (-50 level)
        exit_long = position == 1 and williams_r_aligned[i] > -50
        exit_short = position == -1 and williams_r_aligned[i] < -50
        
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

name = "6h_1d_williamsr_extreme_volume_trend_v1"
timeframe = "6h"
leverage = 1.0