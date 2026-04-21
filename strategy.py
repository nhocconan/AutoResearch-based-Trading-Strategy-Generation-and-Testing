#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume confirmation.
Williams %R identifies overbought/oversold conditions. In trending markets (price above/below 1d EMA34),
extreme readings can signal continuation rather than reversal. Volume confirms momentum.
Designed for 6h timeframe to reduce trade frequency and avoid fee drag, targeting 15-30 trades/year.
Works in bull/bear via trend filter - only takes longs in uptrend, shorts in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for trend filter and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # 14-period lookback
    high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    close_for_wr = df_1d['close'].values
    wr = (high_14 - close_for_wr) / (high_14 - low_14) * -100
    # Handle division by zero when high == low
    wr = np.where((high_14 - low_14) == 0, -50, wr)
    
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    
    # Volume confirmation: volume / 14-period average volume (1d)
    vol_ma_14 = pd.Series(df_1d['volume'].values).rolling(window=14, min_periods=14).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_14
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(wr_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_1d_aligned[i]
        wr_val = wr_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.3  # Volume must be 1.3x average
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80), volume spike, uptrend
            if (wr_val < -80 and 
                vol_ratio > vol_threshold and 
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20), volume spike, downtrend
            elif (wr_val > -20 and 
                  vol_ratio > vol_threshold and 
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R returns to neutral range (-50) or trend reversal
            if position == 1 and (wr_val > -50 or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (wr_val < -50 or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0