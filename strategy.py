#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Donchian channel breakout + volume confirmation + ATR volatility filter.
# Long: Price breaks above 20-period 1d Donchian upper band + volume > 1.3x average volume + ATR(14) > 0.5 * price.
# Short: Price breaks below 20-period 1d Donchian lower band + volume > 1.3x average volume + ATR(14) > 0.5 * price.
# Uses 1d Donchian channels for structural support/resistance, volume for conviction, ATR to avoid low-volatility whipsaws.
# Designed for fewer trades (<50/year) to minimize fee drag and work in both bull/bear regimes via volatility filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on 1d data
    donchian_high = np.full(len(close), np.nan)
    donchian_low = np.full(len(close), np.nan)
    for i in range(20, len(df_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # ATR(14) for volatility filter
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    # Align 1d Donchian levels to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        volatility = atr[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        # Volatility filter: ATR > 0.5 * price (avoid low-volatility chop)
        vol_filter = volatility > 0.5 * price
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume + vol filter
            if (price > upper and volume_confirm and vol_filter):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian + volume + vol filter
            elif (price < lower and volume_confirm and vol_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower Donchian (opposite band)
            if price < lower:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper Donchian (opposite band)
            if price > upper:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Volume_Volatility"
timeframe = "4h"
leverage = 1.0