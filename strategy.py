#!/usr/bin/env python3
"""
6h_WeeklyDonchianBreakout_1dTrend_Volume
Hypothesis: Use weekly Donchian(20) breakout for directional bias, filtered by 1d EMA50 trend and volume spike.
Targets breakouts in strong trends with institutional volume confirmation. Works in bull/bear by following
weekly trend while using daily EMA for noise filtering. Target 15-30 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly Donchian(20) breakout levels ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_20 = align_htf_to_ltf(prices, df_1w, highest_high_20)
    donchian_low_20 = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    
    # Load daily HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_20[i]) or
            np.isnan(donchian_low_20[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        donchian_high = donchian_high_20[i]
        donchian_low = donchian_low_20[i]
        trend_1d = ema_50_1d_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Close breaks above weekly Donchian high + price above 1d EMA50 + volume spike
            if (price_close > donchian_high and
                price_close > trend_1d and
                vol_spike > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below weekly Donchian low + price below 1d EMA50 + volume spike
            elif (price_close < donchian_low and
                  price_close < trend_1d and
                  vol_spike > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses back below/above 1d EMA50
            if position == 1 and price_close < trend_1d:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyDonchianBreakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0