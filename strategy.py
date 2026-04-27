#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d daily Williams %R mean reversion with 1-week EMA trend filter and volume confirmation
# Williams %R below -80 indicates oversold conditions (long), above -20 indicates overbought (short)
# 1-week EMA(34) determines trend direction: only take longs in uptrend, shorts in downtrend
# Volume > 1.5x 20-period average confirms conviction
# Target: 15-25 trades/year to minimize fee decay while capturing mean reversion moves
# Works in both bull and bear markets by aligning with weekly trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate Williams %R on 1d (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    n_1d = len(close_1d)
    
    williams_r = np.full(n_1d, np.nan)
    lookback_period = 14
    
    for i in range(lookback_period - 1, n_1d):
        highest_high = np.max(high_1d[i - lookback_period + 1:i + 1])
        lowest_low = np.min(low_1d[i - lookback_period + 1:i + 1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Calculate EMA on 1w (34-period)
    close_1w = df_1w['close'].values
    ema_period = 34
    ema_1w = np.full(len(close_1w), np.nan)
    
    if len(close_1w) >= ema_period:
        # Calculate initial SMA
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        # Calculate EMA
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] - ema_1w[i - 1]) * multiplier + ema_1w[i - 1]
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Align HTF indicators to 1d
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(lookback_period, vol_period, ema_period)
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Conditions:
        # 1. Williams %R oversold/overbought: < -80 for long, > -20 for short
        # 2. Weekly EMA trend: price above EMA for long bias, below for short bias
        # 3. Volume confirmation: > 1.5x average volume
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        uptrend = price > ema_1w_aligned[i]
        downtrend = price < ema_1w_aligned[i]
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: oversold during uptrend with volume
            if oversold and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: overbought during downtrend with volume
            elif overbought and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) or trend changes
            if williams_r_aligned[i] > -50 or price <= ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) or trend changes
            if williams_r_aligned[i] < -50 or price >= ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WilliamsR_WeeklyEMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0