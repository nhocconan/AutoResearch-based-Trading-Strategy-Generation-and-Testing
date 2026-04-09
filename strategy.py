#!/usr/bin/env python3
# 6h_elder_ray_regime_v5
# Hypothesis: 6h strategy using Elder Ray (Bull/Bear Power) with 1d ADX regime filter.
# In trending markets (ADX>25): go long when Bull Power > 0 and Bear Power < 0.
# In ranging markets (ADX<20): fade extremes - short when Bull Power > 0.3*ATR, long when Bear Power < -0.3*ATR.
# Volume confirmation filters weak signals. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Target: 12-37 trades/year (50-150 over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_regime_v5"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for regime and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (1d)
    close_1d = pd.Series(df_1d['close'].values)
    ema13 = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Elder Ray components (1d)
    bull_power = high_1d - ema13  # Bull Power = High - EMA13
    bear_power = low_1d - ema13   # Bear Power = Low - EMA13
    
    # ATR for volatility normalization (1d)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d.iloc[0]], close_1d.iloc[:-1]].values)) if hasattr(close_1d, 'iloc') else np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d.iloc[0]], close_1d.iloc[:-1]], values)) if hasattr(close_1d, 'iloc') else np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ADX for regime detection (1d)
    plus_dm = np.where((high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])) > (np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d), 
                       np.maximum(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]), 0), 0)
    minus_dm = np.where((np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d) > (high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])), 
                        np.maximum(np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d, 0), 0)
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if adx_aligned[i] > 25:  # Trending regime
                # Exit when Bear Power becomes positive (trend weakening)
                if bear_power_aligned[i] > 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime
                # Exit when price reverts to mean (Bear Power > -0.1*ATR)
                if bear_power_aligned[i] > -0.1 * atr_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx_aligned[i] > 25:  # Trending regime
                # Exit when Bull Power becomes negative (trend weakening)
                if bull_power_aligned[i] < 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime
                # Exit when price reverts to mean (Bull Power < 0.1*ATR)
                if bull_power_aligned[i] < 0.1 * atr_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                if adx_aligned[i] > 25:  # Trending regime
                    # Trend following: long when Bull Power > 0 and Bear Power < 0
                    if bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0:
                        position = 1
                        signals[i] = 0.25
                    # Short when Bear Power > 0 and Bull Power < 0
                    elif bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0:
                        position = -1
                        signals[i] = -0.25
                else:  # Ranging regime
                    # Mean reversion: fade extremes
                    # Short when Bull Power > 0.3*ATR (overbought)
                    if bull_power_aligned[i] > 0.3 * atr_aligned[i]:
                        position = -1
                        signals[i] = -0.25
                    # Long when Bear Power < -0.3*ATR (oversold)
                    elif bear_power_aligned[i] < -0.3 * atr_aligned[i]:
                        position = 1
                        signals[i] = 0.25
    
    return signals