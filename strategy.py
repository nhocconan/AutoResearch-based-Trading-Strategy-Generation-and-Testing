#!/usr/bin/env python3
# 12h_1w_Donchian_Breakout_Trend_1wMA
# Hypothesis: On the 12h timeframe, weekly Donchian channel breakouts (20-period) in the direction of the 1-week EMA trend, with volume confirmation, capture major trends while avoiding whipsaws. The weekly trend filter ensures we only trade in the direction of the higher-timeframe momentum, reducing false signals during corrections. This approach works in both bull and bear markets by capturing sustained moves and exiting when the trend weakens. Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # === 1w data (HTF for Donchian and EMA trend) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # === 1w Donchian channel (20-period) ===
    donchian_high = np.full_like(close_1w, np.nan)
    donchian_low = np.full_like(close_1w, np.nan)
    for i in range(20, len(close_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
    
    # === 1w EMA (34-period) for trend filter ===
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 12h volume ratio for confirmation ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / vol_ma_20_12h
    
    # Align all 1w data to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Donchian and EMA calculations
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_ratio = vol_ratio_12h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA (trend change) OR touches opposite Donchian band
            if price < ema_trend or price < lower:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA (trend change) OR touches opposite Donchian band
            if price > ema_trend or price > upper:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above upper Donchian band with volume, in uptrend (price > weekly EMA)
            if price > upper and vol_ratio > 1.5 and price > ema_trend:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Break below lower Donchian band with volume, in downtrend (price < weekly EMA)
            elif price < lower and vol_ratio > 1.5 and price < ema_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1w_Donchian_Breakout_Trend_1wMA"
timeframe = "12h"
leverage = 1.0