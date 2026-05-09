#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h directional filter and 1d trend filter.
# Uses 4h Donchian channel breakout for direction and 1d EMA50 for trend confirmation.
# Entry on 1h breakout of 4h Donchian(20) with volume confirmation (>1.5x average).
# Designed for low trade frequency (15-30/year) to minimize fee drag in ranging markets.
# Works in bull markets via trend continuation and in bear via mean reversion at extremes.
name = "1h_Donchian20_4hDir_1dEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel (directional filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channel on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian upper and lower bands
    donchian_high_4h = np.full_like(high_4h, np.nan)
    donchian_low_4h = np.full_like(low_4h, np.nan)
    
    for i in range(20, len(high_4h)):
        donchian_high_4h[i] = np.max(high_4h[i-20:i])
        donchian_low_4h[i] = np.min(low_4h[i-20:i])
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need 50 for 1d EMA and 20 for 4h Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        ema_1d = ema_50_1d_aligned[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:max(i,1)]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Price > 4h Donchian high AND price > 1d EMA50 (uptrend) AND volume > 1.5x average
            if close[i] > donchian_high and close[i] > ema_1d and vol > 1.5 * vol_ma:
                signals[i] = 0.20
                position = 1
            # Enter short: Price < 4h Donchian low AND price < 1d EMA50 (downtrend) AND volume > 1.5x average
            elif close[i] < donchian_low and close[i] < ema_1d and vol > 1.5 * vol_ma:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price < 4h Donchian low OR trend reverses (price < 1d EMA50)
            if close[i] < donchian_low or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price > 4h Donchian high OR trend reverses (price > 1d EMA50)
            if close[i] > donchian_high or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals