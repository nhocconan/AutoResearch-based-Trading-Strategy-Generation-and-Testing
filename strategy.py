#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with 12-hour trend filter and volume confirmation.
# Uses 12h Donchian channels for breakout signals, filtered by 12h EMA50 trend and volume spike (>2x EMA20 volume).
# Designed to work in both bull and bear markets by requiring trend alignment and institutional participation.
name = "6h_Donchian20_EMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Donchian and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high over last 20 periods
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 periods
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter: volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ema20)
    
    # Align 12h indicators to 6h timeframe
    donchian_upper_6h = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_6h = align_htf_to_ltf(prices, df_12h, donchian_lower)
    ema_50_12h_6h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 to be valid
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(donchian_upper_6h[i]) or np.isnan(donchian_lower_6h[i]) or
            np.isnan(ema_50_12h_6h[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume spike and above EMA50
            if (price > donchian_upper_6h[i] and vol_spike[i] and price > ema_50_12h_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with volume spike and below EMA50
            elif (price < donchian_lower_6h[i] and vol_spike[i] and price < ema_50_12h_6h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Donchian lower (mean reversion)
            if price < donchian_lower_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Donchian upper (mean reversion)
            if price > donchian_upper_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals