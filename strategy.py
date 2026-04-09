#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR volatility filter + volume confirmation
# Uses 4h Donchian channels for breakout signals, confirmed by volume spike (>1.8x 20-period avg volume)
# Only takes breakouts when 1d ATR(14) is above its 50-period SMA (high volatility regime) to avoid chop
# Position size 0.25 to manage drawdown
# Target: 100-180 total trades over 4 years (25-45/year) to balance edge and fee drag
# Works in both bull/bear: volatility filter ensures we trade during strong moves, avoid ranging markets

name = "4h_1d_donchian_vol_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and its 50-period SMA for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    # ATR(14)
    atr_14 = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if i == 14:
            atr_14[i] = np.mean(tr[1:15])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # ATR(14) 50-period SMA
    atr_sma_50 = np.full(len(df_1d), np.nan)
    for i in range(50, len(df_1d)):
        atr_sma_50[i] = np.mean(atr_14[i-50:i])
    
    # Align 1d ATR and its SMA to 4h timeframe
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_sma_50_4h = align_htf_to_ltf(prices, df_1d, atr_sma_50)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(atr_14_4h[i]) or 
            np.isnan(atr_sma_50_4h[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 1d ATR(14) > its 50-period SMA (high volatility regime)
        vol_filter = atr_14_4h[i] > atr_sma_50_4h[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirm = volume[i] > 1.8 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout with volume confirmation and volatility filter
            if vol_filter and volume_confirm:
                # Long breakout: price closes above upper Donchian band
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below lower Donchian band
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals