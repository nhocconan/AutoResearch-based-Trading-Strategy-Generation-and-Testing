#!/usr/bin/env python3
"""
1h_HTF_MeanReversion_With_Volume_Regime
Hypothesis: Uses 4h RSI and 1d ATR-based channels for mean reversion signals on 1h timeframe.
Only trades during 08-20 UTC session to reduce noise. Uses volume contraction/expansion
to filter false signals. Designed for low trade frequency (15-30/year) to avoid fee drag.
Works in both bull and bear markets by fading extremes when volatility is low.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    
    # Get 4h data for RSI (mean reversion signal)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h RSI(14)
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Get 1d data for ATR-based channels (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d ATR(10) and mean
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    mean_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Upper and lower bands (mean ± 1.5 * ATR)
    upper_band = mean_1d + (1.5 * atr_1d)
    lower_band = mean_1d - (1.5 * atr_1d)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Volume filter: current volume < 20-period MA (volatility contraction)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(upper_band_aligned[i]) or
            np.isnan(lower_band_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Mean reversion signals
        # Long when RSI oversold and price near lower band
        long_signal = (rsi_4h_aligned[i] < 30) and (close[i] <= lower_band_aligned[i] * 1.005)
        # Short when RSI overbought and price near upper band
        short_signal = (rsi_4h_aligned[i] > 70) and (close[i] >= upper_band_aligned[i] * 0.995)
        
        # Volume contraction filter (avoid high volatility breakouts)
        vol_contract = volume[i] < vol_ma_20[i]
        
        # Exit when price returns to mean or RSI normalizes
        long_exit = (close[i] >= mean_1d_aligned[i]) or (rsi_4h_aligned[i] > 50)
        short_exit = (close[i] <= mean_1d_aligned[i]) or (rsi_4h_aligned[i] < 50)
        
        if long_signal and vol_contract and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_signal and vol_contract and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_HTF_MeanReversion_With_Volume_Regime"
timeframe = "1h"
leverage = 1.0