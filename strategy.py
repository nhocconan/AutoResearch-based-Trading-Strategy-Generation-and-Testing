#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation.
# Donchian(20) provides price channel breakout signals; 1w EMA filter ensures trades align with higher-timeframe trend.
# Volume confirmation reduces false signals. Designed for low frequency (target 7-25 trades/year) to minimize fee drag.
# Works in bull/bear by following weekly trend direction.
name = "1d_Donchian20_1wEMA40_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for Donchian calculation (higher frequency for channel)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1h data
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    
    # Donchian upper and lower bands (20-period)
    donchian_high = pd.Series(high_1h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1h, donchian_low)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # 1w EMA(40) for trend filter
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Volume confirmation: volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from alignment)
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_40_1w_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + above 1w EMA40 + volume confirmation
            if (price > donchian_high_aligned[i] and price > ema_40_1w_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below 1w EMA40 + volume confirmation
            elif (price < donchian_low_aligned[i] and price < ema_40_1w_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below Donchian low
            if price < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above Donchian high
            if price > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals