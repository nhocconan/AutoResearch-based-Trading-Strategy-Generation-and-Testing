#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R with 1d EMA34 trend filter and volume spike.
Williams %R identifies overbought/oversold conditions. Buy when %R < -80 (oversold) with
volume spike and uptrend (price > EMA34). Sell when %R > -20 (overbought) with volume
spike and downtrend (price < EMA34). Uses 1d EMA34 for trend filter to avoid counter-trend
trades. Designed for ~15-30 trades/year on 12h to minimize fee drift, works in bull/bear via
trend filter and mean-reversion logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA34 trend filter and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100, period=14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d_arr) / (highest_high - lowest_low)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: volume / 20-period average volume (1d)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_1d_aligned[i]
        williams_r = williams_r_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.5  # Volume must be 1.5x average
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80), volume spike, uptrend
            if (williams_r < -80 and 
                vol_ratio > vol_threshold and 
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20), volume spike, downtrend
            elif (williams_r > -20 and 
                  vol_ratio > vol_threshold and 
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R returns to -50 (mean reversion) or trend reversal
            if position == 1 and (williams_r > -50 or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (williams_r < -50 or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0