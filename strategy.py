#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 1w Donchian breakout with volume confirmation and volatility filter
# Uses weekly Donchian(20) breakout with volume > 1.5x 20-period average
# Enters only when 1d ATR rank < 40 (low volatility environment) to avoid chop
# Exits when price closes opposite weekly Donchian band
# Position size 0.25 to limit drawdown
# Target: 10-25 trades/year per symbol to minimize fee drag

name = "1d_1w_donchian_vol_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donch_high_1w = np.full(len(df_1w), np.nan)
    donch_low_1w = np.full(len(df_1w), np.nan)
    
    for i in range(20, len(df_1w)):
        donch_high_1w[i] = np.max(high_1w[i-20:i])
        donch_low_1w[i] = np.min(low_1w[i-20:i])
    
    # Align weekly Donchian to 1d timeframe (only use completed weekly bars)
    donch_high_1d = align_htf_to_ltf(prices, df_1w, donch_high_1w)
    donch_low_1d = align_htf_to_ltf(prices, df_1w, donch_low_1w)
    
    # Calculate 1d ATR (14-period)
    high_1d = df_1w['high'].values  # Correction: use 1d data for ATR
    low_1d = df_1w['low'].values
    close_1d = df_1w['close'].values
    
    # Actually load 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
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
    
    # ATR percentile rank (200-day lookback)
    atr_rank_1d = np.zeros(len(df_1d))
    for i in range(200, len(df_1d)):
        window = atr_1d[i-200:i]
        atr_rank_1d[i] = np.sum(window < atr_1d[i]) / len(window) * 100
    
    # Align ATR rank to 1d timeframe (already aligned, but for consistency)
    atr_rank_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_rank_1d)
    
    # Volume confirmation: 20-period average on 1d
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
        if (np.isnan(donch_high_1d[i]) or 
            np.isnan(donch_low_1d[i]) or 
            np.isnan(atr_rank_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in low volatility environment (ATR rank < 40 = bottom 40% volatility)
        if atr_rank_1d_aligned[i] >= 40:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly Donchian low
            if close[i] <= donch_low_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly Donchian high
            if close[i] >= donch_high_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above weekly Donchian high with volume confirmation
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (close[i] > donch_high_1d[i] and 
                vol_ratio > 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below weekly Donchian low with volume confirmation
            elif (close[i] < donch_low_1d[i] and 
                  vol_ratio > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals