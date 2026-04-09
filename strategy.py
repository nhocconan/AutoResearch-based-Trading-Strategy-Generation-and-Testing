#!/usr/bin/env python3
# 1d_kama_rsi_chop_v1
# Hypothesis: Daily strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction,
# RSI(14) for momentum confirmation, and Choppiness Index (CHOP) for regime filter.
# Long when: price > KAMA(14,2,30), RSI > 50, and CHOP > 61.8 (ranging market -> mean reversion to mean)
# Short when: price < KAMA(14,2,30), RSI < 50, and CHOP > 61.8 (ranging market -> mean reversion from mean)
# In ranging markets (CHOP > 61.8), price tends to revert to the mean (KAMA), providing edge in both bull and bear regimes.
# Volume confirmation: current volume > 1.3x 20-period average to ensure participation.
# Discrete sizing (±0.25) minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # KAMA parameters
    er_period = 10
    fast_sc = 2
    slow_sc = 30
    
    # Calculate KAMA
    change = np.abs(np.diff(close, n=1))
    change = np.insert(change, 0, 0)  # align length
    
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    volatility = pd.Series(volatility).rolling(window=er_period, min_periods=1).sum().values
    volatility = np.insert(volatility, 0, volatility[0])  # align length
    
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc = np.where(sc > 0, sc, 0.000001)  # avoid division by zero
    
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (CHOP)
    chop_period = 14
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    chop = np.where((max_high - min_low) > 0, 
                    100 * np.log10(atr / (max_high - min_low)) / np.log10(chop_period), 
                    50)
    
    # Align weekly close to daily timeframe
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(close_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA OR RSI < 40 (momentum loss)
            if close[i] < kama[i] or rsi[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA OR RSI > 60 (momentum loss)
            if close[i] > kama[i] or rsi[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and choppy market (CHOP > 61.8 = ranging)
            if volume_confirmed and chop[i] > 61.8:
                # Long: price above KAMA + RSI > 50 (bullish momentum) + weekly uptrend
                if close[i] > kama[i] and rsi[i] > 50 and close_1w_aligned[i] > close_1w_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price below KAMA + RSI < 50 (bearish momentum) + weekly downtrend
                elif close[i] < kama[i] and rsi[i] < 50 and close_1w_aligned[i] < close_1w_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals