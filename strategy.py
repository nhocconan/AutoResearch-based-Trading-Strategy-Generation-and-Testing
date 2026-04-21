#!/usr/bin/env python3
"""
Hypothesis: 1d strategy combining 1-week Donchian breakout with 1-day EMA trend filter and volume confirmation.
Uses weekly Donchian channels (20-period) for breakout direction, filters with daily EMA50 trend, and requires
volume > 1.5x 20-day average for confirmation. Target: 10-25 trades/year to minimize fee drag while capturing
major trend moves. Works in both bull and bear markets by trading with the higher timeframe (weekly) trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Load 1d data ONCE before loop for EMA trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1d volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5  # Volume spike filter
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high + uptrend (price > EMA50) + volume spike
            if (price_close > donch_high and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low + downtrend (price < EMA50) + volume spike
            elif (price_close < donch_low and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through weekly Donchian channel (mean reversion) or trend reversal
            if position == 1 and (price_close < donch_low or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > donch_high or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_DonchianBreakout_1wChannel_1dEMA50_Volume"
timeframe = "1d"
leverage = 1.0