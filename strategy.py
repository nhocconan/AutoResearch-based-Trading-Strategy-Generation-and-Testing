#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# Uses Williams %R to identify overbought/oversold conditions (reversal signals),
# 1d EMA trend filter to ensure alignment with higher timeframe trend,
# and volume > 1.5x average to confirm momentum.
# Works in both bull and bear markets by only taking reversals in the direction of the 1d trend.
# Target: 20-30 trades/year to minimize fee decay while capturing high-probability reversals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA on 1d (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2) / (34 + 1) + ema_34_1d[i-1] * (33) / (34 + 1)
    
    # Williams %R on 4h (14-period)
    williams_r = np.full(n, np.nan)
    lookback = 14
    for i in range(lookback, n):
        highest_high = np.max(high[i-lookback+1:i+1])
        lowest_low = np.min(low[i-lookback+1:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align HTF indicators to 4h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(lookback, vol_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Conditions:
        # 1. Williams %R oversold (< -80) or overbought (> -20)
        # 2. Price above/below 1d EMA for trend alignment
        # 3. Volume confirmation: > 1.5x average volume
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        uptrend = price > ema_34_1d_aligned[i]
        downtrend = price < ema_34_1d_aligned[i]
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
            # Long exit: Williams %R returns to neutral or trend changes
            if williams_r[i] > -50 or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Williams %R returns to neutral or trend changes
            if williams_r[i] < -50 or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsR_EMATrend_Volume"
timeframe = "4h"
leverage = 1.0