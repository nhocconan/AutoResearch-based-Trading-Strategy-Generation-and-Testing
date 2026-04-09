#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and low volatility filter
# Uses 12h Donchian(20) breakout with volume > 1.3x 20-period average
# Enters only when 1d ATR rank < 25 (low volatility environment) to avoid chop
# Exits when price closes opposite Donchian band
# Position size 0.25 to limit drawdown
# Target: 15-30 trades/year per symbol to minimize fee drag

name = "12h_1d_donchian_vol_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donch_high_12h = np.full(len(df_12h), np.nan)
    donch_low_12h = np.full(len(df_12h), np.nan)
    
    for i in range(20, len(df_12h)):
        donch_high_12h[i] = np.max(high_12h[i-20:i])
        donch_low_12h[i] = np.min(low_12h[i-20:i])
    
    # Align 12h Donchian to 12h timeframe (only use completed 12h bars)
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # Calculate 1d ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    atr_1d = np.zeros(len(df_1d))
    atr_1d[0] = tr_1d[0]
    for i in range(1, len(df_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # ATR percentile rank (200-day lookback ~ 10 months)
    atr_rank_1d = np.zeros(len(df_1d))
    for i in range(200, len(df_1d)):
        window = atr_1d[i-200:i]
        atr_rank_1d[i] = np.sum(window < atr_1d[i]) / len(window) * 100
    
    # Align ATR rank to 12h timeframe (only use completed daily bars)
    atr_rank_12h = align_htf_to_ltf(prices, df_1d, atr_rank_1d)
    
    # Volume confirmation: 20-period average on 12h (10 days)
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after ATR rank warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high_12h_aligned[i]) or 
            np.isnan(donch_low_12h_aligned[i]) or 
            np.isnan(atr_rank_12h[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in low volatility environment (ATR rank < 25 = bottom 25% volatility)
        if atr_rank_12h[i] >= 25:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 12h Donchian low
            if close[i] <= donch_low_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h Donchian high
            if close[i] >= donch_high_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 12h Donchian high with volume confirmation
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (close[i] > donch_high_12h_aligned[i] and 
                vol_ratio > 1.3):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 12h Donchian low with volume confirmation
            elif (close[i] < donch_low_12h_aligned[i] and 
                  vol_ratio > 1.3):
                position = -1
                signals[i] = -0.25
    
    return signals