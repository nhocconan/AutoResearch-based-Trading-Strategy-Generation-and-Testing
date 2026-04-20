#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA200 trend filter + volume confirmation
# - Long when price breaks above Donchian high (20) and price > 1d EMA200 and volume > 1.5x average
# - Short when price breaks below Donchian low (20) and price < 1d EMA200 and volume > 1.5x average
# - Exit when price crosses back through Donchian midpoint or trend reverses
# - Uses volatility breakout with trend filter to capture strong moves while avoiding false signals
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA200 calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(200) on 1d timeframe
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 4h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Donchian Channel (20) on 4h timeframe
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Donchian high and low (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: 1.5x average volume
    vol_avg = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_avg * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        ema200 = ema_200_1d_aligned[i]
        vol_thresh = vol_threshold[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + above 1d EMA200 + volume confirmation
            if price > upper and price > ema200 and vol > vol_thresh:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + below 1d EMA200 + volume confirmation
            elif price < lower and price < ema200 and vol > vol_thresh:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint or trend reverses (price < EMA200)
            if price < mid or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint or trend reverses (price > EMA200)
            if price > mid or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_1dEMA200_VolumeFilter"
timeframe = "4h"
leverage = 1.0