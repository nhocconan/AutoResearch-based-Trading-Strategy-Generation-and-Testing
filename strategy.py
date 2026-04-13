#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Squeeze breakout with 1d ADX trend filter and volume confirmation.
# Bollinger Squeeze (low volatility) precedes explosive moves. When bands contract
# (BB width < 20th percentile), a breakout is imminent. Direction determined by:
# 1) Price breakout above/below Bollinger Bands
# 2) 1d ADX > 25 confirming trend strength
# 3) Volume > 2.0x average for confirmation
# Works in both bull/bear markets by capturing volatility expansions.
# Target: 20-50 trades per year (80-200 total over 4 years) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2.0)
    bb_length = 20
    bb_mult = 2.0
    basis = np.zeros(n)
    dev = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    # Calculate SMA and standard deviation
    for i in range(bb_length - 1, n):
        basis[i] = np.mean(close[i - bb_length + 1:i + 1])
        dev[i] = np.std(close[i - bb_length + 1:i + 1])
        upper[i] = basis[i] + bb_mult * dev[i]
        lower[i] = basis[i] - bb_mult * dev[i]
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper - lower) / basis
    # Percentile lookback for squeeze threshold (50 periods)
    bb_width_percentile = np.full(n, np.nan)
    for i in range(50, n):
        bb_width_percentile[i] = np.percentile(bb_width[i-50:i], 20)  # 20th percentile
    
    # Squeeze condition: BB width < 20th percentile of last 50 periods
    squeeze = np.zeros(n, dtype=bool)
    for i in range(50, n):
        if not np.isnan(bb_width[i]) and not np.isnan(bb_width_percentile[i]):
            squeeze[i] = bb_width[i] < bb_width_percentile[i]
    
    # 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14)
    adx_period = 14
    tr = np.zeros(len(close_1d))
    plus_dm = np.zeros(len(close_1d))
    minus_dm = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
            
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
    
    # Smoothed values
    atr = np.zeros(len(close_1d))
    plus_di = np.zeros(len(close_1d))
    minus_di = np.zeros(len(close_1d))
    dx = np.zeros(len(close_1d))
    
    # Initial values
    if len(close_1d) >= adx_period:
        atr[adx_period-1] = np.mean(tr[1:adx_period])
        plus_di[adx_period-1] = (np.sum(plus_dm[1:adx_period]) / atr[adx_period-1]) * 100
        minus_di[adx_period-1] = (np.sum(minus_dm[1:adx_period]) / atr[adx_period-1]) * 100
        
        if plus_di[adx_period-1] + minus_di[adx_period-1] != 0:
            dx[adx_period-1] = (abs(plus_di[adx_period-1] - minus_di[adx_period-1]) / 
                               (plus_di[adx_period-1] + minus_di[adx_period-1])) * 100
    
    # Smooth subsequent values
    for i in range(adx_period, len(close_1d)):
        atr[i] = (atr[i-1] * (adx_period - 1) + tr[i]) / adx_period
        plus_di[i] = (plus_di[i-1] * (adx_period - 1) + plus_dm[i]) / atr[i] * 100
        minus_di[i] = (minus_di[i-1] * (adx_period - 1) + minus_dm[i]) / atr[i] * 100
        
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # ADX calculation
    adx = np.zeros(len(close_1d))
    if len(close_1d) >= 2 * adx_period - 1:
        adx[2*adx_period-2] = np.mean(dx[adx_period-1:2*adx_period-1])
        for i in range(2*adx_period-1, len(close_1d)):
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Average volume (20-period = 10 hours) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(basis[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        adx_val = adx_1d_aligned[i]
        is_squeeze = squeeze[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        # Only trade when coming out of a squeeze (volatility expansion)
        if not is_squeeze:
            continue
            
        if position == 0:
            # Long: Price breaks above upper Bollinger Band + ADX > 25 + volume confirmation
            if (price > upper[i] and
                adx_val > 25 and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower Bollinger Band + ADX > 25 + volume confirmation
            elif (price < lower[i] and
                  adx_val > 25 and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price returns to middle Bollinger Band or ADX weakens
            if (price < basis[i] or
                adx_val < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price returns to middle Bollinger Band or ADX weakens
            if (price > basis[i] or
                adx_val < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_BollingerSqueeze_ADX_Volume"
timeframe = "4h"
leverage = 1.0