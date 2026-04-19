#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d EMA trend filter.
# Long when: price breaks above upper Donchian band (20-period high) with volume > 1.5x 20-period average and price > 1d EMA50
# Short when: price breaks below lower Donchian band (20-period low) with volume > 1.5x 20-period average and price < 1d EMA50
# Exit when price returns to the middle of the Donchian channel (mean of upper and lower bands) or reverses to opposite band.
# Uses Donchian channels for trend following breakouts, volume to confirm breakouts,
# and higher timeframe trend to avoid counter-trend trades. Designed for ~12-20 trades/year per symbol.
name = "12h_Donchian20_Volume_EMA50Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period) on 12h timeframe
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Get current levels
        upper = high_max_20[i]
        lower = low_min_20[i]
        mid = donchian_mid[i]
        ema_50 = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price > upper band with volume confirmation and uptrend
            if price > upper and vol > 1.5 * vol_ma and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower band with volume confirmation and downtrend
            elif price < lower and vol > 1.5 * vol_ma and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band or breaks below lower band (reversal)
            if price <= mid or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band or breaks above upper band (reversal)
            if price >= mid or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals