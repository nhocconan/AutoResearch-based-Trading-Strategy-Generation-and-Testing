#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 14-day Donchian breakout with 100-day EMA trend filter and volume confirmation.
In uptrend (price > EMA100), buy breakouts above Donchian high; in downtrend (price < EMA100), sell breakdowns below Donchian low.
Donchian channels capture breakouts from consolidation, EMA100 filters for strong trend alignment, volume confirms breakout strength.
Works in bull markets (buy breakouts) and bear markets (sell breakdowns). Target: 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 100-day EMA data ONCE before loop for trend filter
    df_100d = get_htf_data(prices, '1d')
    if len(df_100d) < 100:
        return np.zeros(n)
    
    # 100-day EMA for trend filter
    close_1d = df_100d['close'].values
    ema_100 = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_aligned = align_htf_to_ltf(prices, df_100d, ema_100)
    
    # Load 14-day Donchian data ONCE before loop
    df_14d = get_htf_data(prices, '1d')
    if len(df_14d) < 14:
        return np.zeros(n)
    
    # 14-day Donchian channel
    high_1d = df_14d['high'].values
    low_1d = df_14d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    donchian_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_14d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_14d, donchian_low)
    
    # 4h volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_100_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_100_aligned[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5  # Volume spike filter
        
        if position == 0:
            # Enter long: price breaks above Donchian high + uptrend (price > EMA100) + volume spike
            if (price_close > donch_high and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + downtrend (price < EMA100) + volume spike
            elif (price_close < donch_low and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (price crosses EMA100 in opposite direction)
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

name = "4h_Donchian14_100dEMA_Volume"
timeframe = "4h"
leverage = 1.0