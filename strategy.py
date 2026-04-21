#!/usr/bin/env python3
"""
6h_Aggressive_Trend_Follower
Hypothesis: Use 6h Donchian channel breakout with volume confirmation and 1d EMA trend filter to capture strong trends in BTC/ETH. Designed to work in both bull and bear markets by following higher timeframe trend while using price breakouts and volume for entry. Target 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h Donchian channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].values[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike > 1.5 + price above 1d EMA34
            if (price_close > highest_high[i] and 
                vol_spike > 1.5 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike > 1.5 + price below 1d EMA34
            elif (price_close < lowest_low[i] and 
                  vol_spike > 1.5 and 
                  price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses back through Donchian middle
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if position == 1 and price_close < donchian_mid:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Aggressive_Trend_Follower"
timeframe = "6h"
leverage = 1.0