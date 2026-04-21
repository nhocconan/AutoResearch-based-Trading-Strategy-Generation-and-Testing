#!/usr/bin/env python3
"""
4h_Donchian_Breakout_1dTrend_Volume
Hypothesis: Use Donchian channel breakout (20-period) on 4h with 1d EMA50 trend filter and volume confirmation. Designed to capture strong directional moves in both bull and bear markets by combining price breakouts with higher timeframe trend alignment and volume validation. Target 20-50 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Donchian channel on 4h (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation: 20-period volume average on 4h ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        trend_1d = ema_50_1d_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper + volume spike + price above 1d EMA50
            if (price_high > upper_channel and 
                vol_spike > 1.8 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + volume spike + price below 1d EMA50
            elif (price_low < lower_channel and 
                  vol_spike > 1.8 and 
                  price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to opposite Donchian band
            if position == 1 and price_low < lower_channel:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_high > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0