#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume spike confirmation
    # Williams %R identifies overbought/oversold conditions; 1w EMA200 provides major trend filter
    # Volume spike confirms institutional interest at reversal points
    # Target: 60-100 trades over 4 years (15-25/year) for low fee drag
    # Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 1d data for Williams %R calculation (more responsive than 6h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1w EMA200 for major trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Williams %R(14) on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d volume average (20-period) for spike confirmation
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        volume_spike = volume_1d[i] > 2.0 * vol_avg_20_aligned[i]
        
        # Williams %R conditions: oversold < -80, overbought > -20
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Trend filter: only trade counter to extreme %R in direction of 1w trend
        # In uptrend (price > EMA200): look for oversold to go long
        # In downtrend (price < EMA200): look for overbought to go short
        trend_filter_long = close[i] > ema200_1w_aligned[i]
        trend_filter_short = close[i] < ema200_1w_aligned[i]
        
        # Entry conditions
        enter_long = oversold and volume_spike and trend_filter_long
        enter_short = overbought and volume_spike and trend_filter_short
        
        # Exit conditions: Williams %R returns to neutral zone (-50) or opposite extreme
        exit_long = position == 1 and (williams_r_aligned[i] > -50 or williams_r_aligned[i] < -90)
        exit_short = position == -1 and (williams_r_aligned[i] < -50 or williams_r_aligned[i] > -10)
        
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

name = "6h_1w_williams_r_meanreversion_volume_v1"
timeframe = "6h"
leverage = 1.0