#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 12h Donchian breakout with 1d EMA50 trend filter and volume confirmation.
In uptrend (price > EMA50), buy breakouts above Donchian upper band; in downtrend (price < EMA50), sell breakdowns below Donchian lower band.
Donchian channels provide clear breakout levels, EMA50 filters trend alignment, volume confirms strength.
Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year) with low turnover to minimize fee drag.
Works in bull markets via trend-following breakouts and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h Donchian channels (20-period high/low)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 12h volume confirmation (volume spike > 1.8x 30-period average)
    vol_ma_30 = pd.Series(df_12h['volume'].values).rolling(window=30, min_periods=30).mean().values
    vol_ma_30_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_30)
    vol_ratio = df_12h['volume'].values / vol_ma_30_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.8  # Volume spike filter for quality
        
        if position == 0:
            # Enter long: price breaks above 12h Donchian high + uptrend + volume spike
            if (price_close > donchian_high_aligned[i] and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 12h Donchian low + downtrend + volume spike
            elif (price_close < donchian_low_aligned[i] and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (price crosses EMA50 in opposite direction)
            if position == 1 and price_close < ema_trend:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_DonchianBreakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0