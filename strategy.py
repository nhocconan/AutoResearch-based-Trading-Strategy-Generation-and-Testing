#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_12h
Hypothesis: Donchian(20) breakouts on 4h with volume confirmation (>1.5x 20-bar avg) and 12h EMA50 trend filter. Captures breakouts with institutional interest in trending markets. Designed for low trade frequency (<50/year) to minimize fee drag. Works in bull/bear by following 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h trend filter: 50-period EMA ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Donchian channels (20-period) on 4h ===
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
    
    for i in range(60, n):  # Start after EMA and Donchian warmup
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_12h = ema_50_12h_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Break above Donchian high + volume spike + above 12h EMA50
            if (price_close > upper and 
                vol_spike > 1.5 and 
                price_close > trend_12h):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + volume spike + below 12h EMA50
            elif (price_close < lower and 
                  vol_spike > 1.5 and 
                  price_close < trend_12h):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to opposite Donchian band
            if position == 1 and price_close < lower:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > upper:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_12h"
timeframe = "4h"
leverage = 1.0