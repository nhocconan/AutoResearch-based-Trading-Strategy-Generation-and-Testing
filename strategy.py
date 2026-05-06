#!/usr/bin/env python3
# 4h_1dDonchian_20_1dATR14_Volume_Trend
# Uses daily Donchian channels (20-period) for breakout signals with daily ATR(14) filter
# and 4h volume confirmation. Trades in the direction of the daily trend (price above/below EMA50).
# Designed to work in both bull and bear markets by following the daily trend.
# Target: 75-200 total trades over 4 years (19-50/year) with 0.25 position sizing.

name = "4h_1dDonchian_20_1dATR14_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Upper band: highest high of last 20 days
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 days
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    upper_20_4h = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_4h = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily ATR(14) for volatility filter
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 4h volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(atr_14_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper Donchian with uptrend and volume
            if close[i] > upper_20_4h[i] and close[i] > ema_50_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with downtrend and volume
            elif close[i] < lower_20_4h[i] and close[i] < ema_50_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to EMA50 or breaks below lower Donchian
            if close[i] < ema_50_4h[i] or close[i] < lower_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to EMA50 or breaks above upper Donchian
            if close[i] > ema_50_4h[i] or close[i] > upper_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf