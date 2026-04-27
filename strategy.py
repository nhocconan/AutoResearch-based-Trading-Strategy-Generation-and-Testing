#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot-based reversal strategy with 1d trend filter and volume confirmation
# Uses daily pivot levels (R1, S1) from 1d data for mean-reversion entries when price touches these levels,
# filtered by 1d EMA34 trend direction and volume spikes. Works in both bull and bear markets
# by taking counter-trend reversals at key levels during pullbacks, with trend filter ensuring
# we only trade reversals in the direction of higher timeframe momentum.
# Target: 20-30 trades/year to minimize fee decay while capturing high-probability reversals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d for trend
    close_1d = df_1d['close'].values
    ema_len = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_len:
        multiplier = 2 / (ema_len + 1)
        ema_1d[ema_len-1] = np.mean(close_1d[:ema_len])
        for i in range(ema_len, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate Camarilla pivot levels (R1, S1) from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.roll(close_1d, 1)
    close_1d_shifted[0] = np.nan  # First day has no previous close
    
    # Camarilla R1 = Close + (High - Low) * 1.1/12
    # Camarilla S1 = Close - (High - Low) * 1.1/12
    camarilla_r1 = close_1d_shifted + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d_shifted - (high_1d - low_1d) * 1.1 / 12
    
    # Align indicators to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 20-period average volume on 12h for spike detection
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    vol_period = 20
    vol_ma_12h = np.full(len(vol_12h), np.nan)
    for i in range(vol_period, len(vol_12h)):
        vol_ma_12h[i] = np.mean(vol_12h[i-vol_period:i])
    
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(35, 25) + 20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_12h_aligned[i] if vol_ma_12h_aligned[i] > 0 else 0
        
        # Volume confirmation: at least 1.8x average volume
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long: Price touches S1 level with uptrend and volume (mean reversion long)
            if price <= camarilla_s1_aligned[i] and price > ema_1d_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: Price touches R1 level with downtrend and volume (mean reversion short)
            elif price >= camarilla_r1_aligned[i] and price < ema_1d_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price reaches midpoint (mean reversion complete) or trend reversal
            midpoint = (camarilla_r1_aligned[i] + camarilla_s1_aligned[i]) / 2
            if price >= midpoint or price < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price reaches midpoint or trend reversal
            midpoint = (camarilla_r1_aligned[i] + camarilla_s1_aligned[i]) / 2
            if price <= midpoint or price > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1S1_1dEMA34_Volume_MeanReversion"
timeframe = "12h"
leverage = 1.0