#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1d Williams %R for overbought/oversold conditions,
# 4h EMA(21) for trend direction, and volume confirmation to filter false signals.
# Long when: 1d Williams %R < -80 (oversold), price > 4h EMA(21), volume > 1.5x average.
# Short when: 1d Williams %R > -20 (overbought), price < 4h EMA(21), volume > 1.5x average.
# Uses 1d Williams %R as a contrarian signal with trend filter to avoid counter-trend trades.
# Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
# Target: 20-50 trades per year to minimize fee drag while capturing meaningful moves.

name = "4h_1dWilliamsR_EMA21_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 1d Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    wr = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    wr_oversold = wr < -80
    wr_overbought = wr > -20
    
    # 4h EMA(21)
    ema_21 = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1d Williams %R to 4h
    wr_oversold_aligned = align_htf_to_ltf(prices, df_1d, wr_oversold.astype(float))
    wr_overbought_aligned = align_htf_to_ltf(prices, df_1d, wr_overbought.astype(float))
    # Align 4h EMA to 4h
    ema_21_aligned = align_htf_to_ltf(prices, df_4h, ema_21)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 40  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr_oversold_aligned[i]) or np.isnan(wr_overbought_aligned[i]) or
            np.isnan(ema_21_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d Williams %R oversold, price > 4h EMA(21), volume spike
            if (wr_oversold_aligned[i] and
                close[i] > ema_21_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: 1d Williams %R overbought, price < 4h EMA(21), volume spike
            elif (wr_overbought_aligned[i] and
                  close[i] < ema_21_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: Williams %R overbought, price < EMA, or max 20 bars held
            if (wr_overbought_aligned[i] or 
                close[i] < ema_21_aligned[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R oversold, price > EMA, or max 20 bars held
            if (wr_oversold_aligned[i] or 
                close[i] > ema_21_aligned[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals