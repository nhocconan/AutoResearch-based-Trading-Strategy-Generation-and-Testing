#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 1d Donchian breakout with volume confirmation and ATR filter
# Uses 1d Donchian(20) breakout with volume > 1.5x 24-period average (4h)
# Enters only when 1d ATR rank < 30 (low volatility) to avoid chop
# Exits when price closes opposite Donchian band
# Position size 0.25 to limit drawdown
# Target: 20-40 trades/year per symbol to minimize fee drag

name = "4h_1d_donchian_vol_filter_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donch_high_1d = np.full(len(df_1d), np.nan)
    donch_low_1d = np.full(len(df_1d), np.nan)
    
    for i in range(20, len(df_1d)):
        donch_high_1d[i] = np.max(high_1d[i-20:i])
        donch_low_1d[i] = np.min(low_1d[i-20:i])
    
    # Align 1d Donchian to 4h timeframe (only use completed daily bars)
    donch_high_4h = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_4h = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Calculate 1d ATR (14-period)
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
    
    # ATR percentile rank (252-day lookback ~ 1 year)
    atr_rank_1d = np.zeros(len(df_1d))
    for i in range(252, len(df_1d)):
        window = atr_1d[i-252:i]
        atr_rank_1d[i] = np.sum(window < atr_1d[i]) / len(window) * 100
    
    # Align ATR rank to 4h timeframe (only use completed daily bars)
    atr_rank_4h = align_htf_to_ltf(prices, df_1d, atr_rank_1d)
    
    # Volume confirmation: 24-period average on 4h (6 days)
    vol_ma_24 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma_24[i] = vol_sum / 24
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(252, n):  # Start after ATR rank warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high_4h[i]) or 
            np.isnan(donch_low_4h[i]) or 
            np.isnan(atr_rank_4h[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in low volatility environment (ATR rank < 30 = bottom 30% volatility)
        if atr_rank_4h[i] >= 30:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 1d Donchian low
            if close[i] <= donch_low_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 1d Donchian high
            if close[i] >= donch_high_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 1d Donchian high with volume confirmation
            vol_ratio = volume[i] / vol_ma_24[i] if vol_ma_24[i] > 0 else 0
            if (close[i] > donch_high_4h[i] and 
                vol_ratio > 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 1d Donchian low with volume confirmation
            elif (close[i] < donch_low_4h[i] and 
                  vol_ratio > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals