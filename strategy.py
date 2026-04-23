#!/usr/bin/env python3
"""
Hypothesis: 1h Mean Reversion with 4h/1d Regime Filter
In ranging markets (CHOP > 61.8), price reverts to the 4h VWAP. 
In trending markets (CHOP < 38.2), follow 1d EMA50 direction.
Volume spike confirms genuine moves. Session filter (08-20 UTC) reduces noise.
Designed for 1h timeframe with tight entry conditions to avoid fee drift.
Target: 20-50 trades/year per symbol.
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
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for regime and VWAP
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Choppiness Index on 4h
    atr_4h = pd.Series(
        np.maximum(
            np.maximum(df_4h['high'].values - df_4h['low'].values,
                       np.abs(df_4h['high'].values - np.roll(df_4h['close'].values, 1))),
            np.abs(df_4h['low'].values - np.roll(df_4h['close'].values, 1))
        )
    ).rolling(window=14, min_periods=14).mean()
    
    true_range_sum = atr_4h.rolling(window=14, min_periods=14).sum().values
    highest_high = df_4h['high'].values.rolling(window=14, min_periods=14).max()
    lowest_low = df_4h['low'].values.rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(true_range_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop.values)
    
    # 4h VWAP (typical price * volume) / volume
    typical_price_4h = (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3
    vwap_4h = (typical_price_4h * df_4h['volume'].values).cumsum() / df_4h['volume'].values.cumsum()
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           np.isnan(chop_aligned[i]) or np.isnan(vwap_4h_aligned[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions
            if chop_aligned[i] > 61.8:  # ranging market
                # Mean reversion: price < VWAP
                if close[i] < vwap_4h_aligned[i] and volume[i] > 2.0 * vol_ma[i]:
                    signals[i] = 0.20
                    position = 1
            else:  # trending market (CHOP < 38.2)
                # Follow 1d EMA50 trend
                if close[i] > ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_ma[i]:
                    signals[i] = 0.20
                    position = 1
            
            # Short conditions
            if chop_aligned[i] > 61.8:  # ranging market
                # Mean reversion: price > VWAP
                if close[i] > vwap_4h_aligned[i] and volume[i] > 2.0 * vol_ma[i]:
                    signals[i] = -0.20
                    position = -1
            else:  # trending market
                # Follow 1d EMA50 trend (short)
                if close[i] < ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_ma[i]:
                    signals[i] = -0.20
                    position = -1
        else:
            # Exit conditions
            exit_signal = False
            if position == 1:
                # Exit long: price > VWAP (in ranging) OR price < EMA50 (in trending)
                if chop_aligned[i] > 61.8:
                    if close[i] > vwap_4h_aligned[i]:
                        exit_signal = True
                else:
                    if close[i] < ema_50_1d_aligned[i]:
                        exit_signal = True
            elif position == -1:
                # Exit short: price < VWAP (in ranging) OR price > EMA50 (in trending)
                if chop_aligned[i] > 61.8:
                    if close[i] < vwap_4h_aligned[i]:
                        exit_signal = True
                else:
                    if close[i] > ema_50_1d_aligned[i]:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_MeanReversion_Regime_VWAP_EMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0