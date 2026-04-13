#!/usr/bin/env python3
"""
1d_1w_Keltner_Channel_Breakout_With_Volume
Hypothesis: Weekly Keltner Channels provide robust dynamic support/resistance. 
Breakouts above upper channel or below lower channel with volume expansion and 
trend alignment (using 200-day EMA) capture institutional moves. The weekly 
timeframe filters noise, while daily execution improves timing. Works in bull 
markets (trend continuation) and bear markets (mean reversion at extremes) by 
requiring both breakout and trend confirmation. Targets 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Keltner Channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w < 2):
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Keltner Channel (20-period EMA, 2x ATR)
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_1w = pd.Series(high_1w - low_1w).rolling(window=20, min_periods=20).mean().values
    upper_1w = ema_20_1w + 2 * atr_1w
    lower_1w = ema_20_1w - 2 * atr_1w
    
    # Align weekly Keltner levels to daily
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily EMA200 for trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume expansion: current volume > 1.5x 50-day average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean()
    volume_expansion = volume > (vol_ma_50 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_200[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Breakout above weekly upper Keltner Channel
        # 2. Volume expansion
        # 3. Above daily EMA200 for trend alignment
        breakout_long = (close[i] > upper_1w_aligned[i]) and volume_expansion[i]
        long_condition = breakout_long and (close[i] > ema_200[i])
        
        # Short conditions:
        # 1. Breakdown below weekly lower Keltner Channel
        # 2. Volume expansion
        # 3. Below daily EMA200 for trend alignment
        breakdown_short = (close[i] < lower_1w_aligned[i]) and volume_expansion[i]
        short_condition = breakdown_short and (close[i] < ema_200[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_Keltner_Channel_Breakout_With_Volume"
timeframe = "1d"
leverage = 1.0