#!/usr/bin/env python3
# 4h_donchian_12h_volatility_v1
# Hypothesis: 4h Donchian(20) breakout with 12h ATR-based volatility filter and volume confirmation.
# Long: price breaks above Donchian(20) high + volume > 1.5x 20-period average + 12h ATR(14) > 0.8x 50-period ATR(14) EMA
# Short: price breaks below Donchian(20) low + volume > 1.5x 20-period average + 12h ATR(14) > 0.8x 50-period ATR(14) EMA
# Exit: opposing Donchian breakout or ATR drops below 0.5x 50-period ATR EMA
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_12h_volatility_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for ATR calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR(14) on 12h data
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR EMA(50) on 12h data
    atr_ema_12h = pd.Series(atr_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h ATR and ATR EMA to 4h timeframe
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    atr_ema_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ema_12h)
    
    # Donchian channels (20-period) on 4h data
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_12h_aligned[i]) or
            np.isnan(atr_ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        volatility_filter = atr_12h_aligned[i] > 0.8 * atr_ema_12h_aligned[i]
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR volatility drops too low
            if close[i] < donch_low[i] or atr_12h_aligned[i] < 0.5 * atr_ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR volatility drops too low
            if close[i] > donch_high[i] or atr_12h_aligned[i] < 0.5 * atr_ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume and volatility confirmation
            if close[i] > donch_high[i] and volume_confirmed and volatility_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume and volatility confirmation
            elif close[i] < donch_low[i] and volume_confirmed and volatility_filter:
                position = -1
                signals[i] = -0.25
    
    return signals