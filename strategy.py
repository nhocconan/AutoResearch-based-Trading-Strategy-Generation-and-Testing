#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ATR regime filter
# Uses 4h Donchian channels for breakout signals, confirmed by 12h volume > 1.8x 20-period average
# Only trades when 12h ATR rank < 40 (low-moderate volatility) to avoid whipsaws in chop
# Exits when price closes opposite Donchian level (20-period low for longs, high for shorts)
# Position size 0.25 to limit drawdown
# Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag
# Works in both bull/bear: Donchian provides structure, ATR filter avoids false breakouts in ranging markets

name = "4h_12h_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for volume and ATR
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ATR (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    vol_12h = df_12h['volume'].values
    
    tr_12h = np.zeros(len(df_12h))
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(df_12h)):
        tr0 = high_12h[i] - low_12h[i]
        tr1 = abs(high_12h[i] - close_12h[i-1])
        tr2 = abs(low_12h[i] - close_12h[i-1])
        tr_12h[i] = max(tr0, tr1, tr2)
    
    atr_12h = np.zeros(len(df_12h))
    atr_12h[0] = tr_12h[0]
    for i in range(1, len(df_12h)):
        atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    # ATR percentile rank (100-period lookback ~ 50 days on 12h)
    atr_rank_12h = np.zeros(len(df_12h))
    for i in range(100, len(df_12h)):
        window = atr_12h[i-100:i]
        atr_rank_12h[i] = np.sum(window < atr_12h[i]) / len(window) * 100
    
    # 12h volume 20-period average
    vol_ma_20_12h = np.zeros(len(df_12h))
    vol_sum = 0.0
    for i in range(len(df_12h)):
        vol_sum += vol_12h[i]
        if i >= 20:
            vol_sum -= vol_12h[i-20]
        if i >= 19:
            vol_ma_20_12h[i] = vol_sum / 20
    
    # Align 12h data to 4h timeframe
    atr_rank_4h = align_htf_to_ltf(prices, df_12h, atr_rank_12h)
    vol_ma_20_4h = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:
            start_idx = i - 19
            high_max[i] = np.max(high[start_idx:i+1])
            low_min[i] = np.min(low[start_idx:i+1])
            donchian_high[i] = high_max[i]
            donchian_low[i] = low_min[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr_rank_4h[i]) or 
            np.isnan(vol_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in low-moderate volatility environment (ATR rank < 40)
        if atr_rank_4h[i] >= 40:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian low (20-period)
            if close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian high (20-period)
            if close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation: current 12h volume > 1.8x 20-period average
            vol_ratio = vol_ma_20_4h[i]
            if vol_ratio > 0:
                vol_ratio = volume[i] / vol_ma_20_4h[i]
            else:
                vol_ratio = 0
            
            # Enter long: price closes above 4h Donchian high with volume confirmation
            if (close[i] > donchian_high[i] and 
                vol_ratio > 1.8):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 4h Donchian low with volume confirmation
            elif (close[i] < donchian_low[i] and 
                  vol_ratio > 1.8):
                position = -1
                signals[i] = -0.25
    
    return signals