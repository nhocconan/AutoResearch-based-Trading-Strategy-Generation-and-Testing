#!/usr/bin/env python3
"""
12h_Donchian20_1dTrend_Volume
Hypothesis: Use 12h Donchian(20) breakouts with 1d EMA200 trend filter and volume confirmation.
This strategy captures strong trend continuations in both bull and bear markets by combining
price breakouts with higher timeframe trend and volume confirmation. Target 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === 1d trend filter: 200-period EMA ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 12h Donchian(20) - using 1d data (20 days = ~20*24/12 = 40 periods in 12h) ===
    # Since we don't have direct 12h data, we'll use 1d data and scale appropriately
    # 20 periods in 12h = 20 * (12/24) = 10 days in 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    donchian_low = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === Volume confirmation: 20-period volume average on 1d ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Current 12h volume (approximated from 1h data or use 1d volume scaled)
    # For simplicity, we'll use the 1d volume as proxy for 12h volume strength
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_200_1d_aligned[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        vol_current = volume[i]
        
        # Volume spike condition: current volume > 1.5 * 20-period average
        vol_spike = vol_current > (1.5 * vol_ma) if vol_ma > 0 else False
        
        if position == 0:
            # Long: Price breaks above Donchian high + above 1d EMA200 + volume spike
            if (price_close > upper_band and 
                price_close > trend_1d and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below 1d EMA200 + volume spike
            elif (price_close < lower_band and 
                  price_close < trend_1d and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to the middle of the Donchian channel
            mid_band = (upper_band + lower_band) / 2
            if position == 1 and price_close < mid_band:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > mid_band:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0