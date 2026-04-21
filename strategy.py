#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1-day Williams %R (momentum oscillator) and 1-day EMA(34) trend filter.
Williams %R identifies overbought/oversold conditions, while EMA(34) defines the trend direction.
In both bull and bear markets, buying oversold dips in uptrends and selling overbought rallies in downtrends
provides edge. Volume confirmation and 1-week trend filter reduce false signals. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1-day data ONCE before loop for Williams %R and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # 1-day Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    
    for i in range(13, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-13:i+1])
        lowest_low[i] = np.min(low_1d[i-13:i+1])
    
    willr = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        if highest_high[i] != lowest_low[i]:
            willr[i] = (highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            willr[i] = -50  # neutral when range is zero
    
    # 1-day EMA(34) for trend
    ema_34 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34[i] = (close_1d[i] * 2 / 35) + (ema_34[i-1] * 33 / 35)
    
    # 1-week EMA(10) for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_1w_10 = np.full_like(df_1w['close'].values, np.nan)
    if len(df_1w) >= 10 and len(df_1w['close']) >= 10:
        close_1w = df_1w['close'].values
        ema_1w_10[9] = np.mean(close_1w[:10])
        for i in range(10, len(close_1w)):
            ema_1w_10[i] = (close_1w[i] * 2 / 11) + (ema_1w_10[i-1] * 9 / 11)
    
    # Align HTF indicators to 12h timeframe
    willr_12h = align_htf_to_ltf(prices, df_1d, willr)
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_1w_10_12h = align_htf_to_ltf(prices, df_1w, ema_1w_10)
    
    # Volume confirmation: 12h volume / 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if indicators not ready
        if (np.isnan(willr_12h[i]) or np.isnan(ema_34_12h[i]) or 
            np.isnan(ema_1w_10_12h[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5  # Volume spike filter
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) + price above 1d EMA(34) + 1w EMA up + volume
            if (willr_12h[i] < -80 and 
                price_close > ema_34_12h[i] and 
                ema_1w_10_12h[i] > ema_1w_10_12h[i-1] and  # 1w EMA rising
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20) + price below 1d EMA(34) + 1w EMA down + volume
            elif (willr_12h[i] > -20 and 
                  price_close < ema_34_12h[i] and 
                  ema_1w_10_12h[i] < ema_1w_10_12h[i-1] and  # 1w EMA falling
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R returns to neutral zone (-50) or opposite extreme
            if position == 1 and (willr_12h[i] > -50 or willr_12h[i] > -20):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (willr_12h[i] < -50 or willr_12h[i] < -80):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_EMA34_Trend_Filter"
timeframe = "12h"
leverage = 1.0