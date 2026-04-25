#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dATR_Trend_VolumeRegime
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter, volume confirmation, and choppiness regime filter.
Goes long when price breaks above R1 with 1d uptrend, volume > 1.3x 20-period average, and chop < 61.8 (trending regime).
Short when price breaks below S1 with 1d downtrend, volume > 1.3x 20-period average, and chop < 61.8.
Exits on close below/above opposite Camarilla level or trend reversal.
Uses discrete sizing (0.25) to minimize fees. Target: 20-40 trades/year.
Works in bull via breakouts with trend, in bear via mean reversion at extremes with regime filter.
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
    
    # Get 1d data for Camarilla calculations and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today using yesterday's OHLC
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])  # yesterday's close
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])   # yesterday's high
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])     # yesterday's low
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 0.275 * camarilla_range
    s1 = prev_close - 0.275 * camarilla_range
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.3 * vol_ma_20)
    
    # Choppiness regime filter (using 14-period)
    # CHOP > 61.8 = ranging/choppy (avoid), CHOP < 38.2 = strong trend
    # We use CHOP < 61.8 to allow trending regimes only
    atr_14 = pd.Series(np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ])).rolling(window=14, min_periods=14).mean().values
    atr_14 = np.concatenate([[np.nan], atr_14])  # align length
    
    true_range = np.maximum.reduce([
        high - low,
        np.abs(high - np.concatenate([[close[0]], close[:-1]])),
        np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    ])
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    highest_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop_denom = highest_14 - lowest_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop = 100 * np.log10(sum(pd.Series(atr_14).rolling(window=14, min_periods=1).sum()) / 14 / chop_denom) / np.log10(10)
    # Simplified Choppiness Index calculation
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / 14 / chop_denom) / np.log10(10)
    chop = np.where(chop_denom == 0, 50, chop)  # default to neutral when no range
    chop_regime = chop < 61.8  # trending regime allowed
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R1, 1d uptrend (price > EMA50), volume spike, trending regime
            long_signal = (close[i] > r1_aligned[i]) and (close[i] > ema_50_1d_aligned[i]) and vol_spike[i] and chop_regime[i]
            # Short: price breaks below S1, 1d downtrend (price < EMA50), volume spike, trending regime
            short_signal = (close[i] < s1_aligned[i]) and (close[i] < ema_50_1d_aligned[i]) and vol_spike[i] and chop_regime[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price closes below S1 (mean reversion) or 1d trend turns down
            exit_signal = (close[i] < s1_aligned[i]) or (close[i] < ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price closes above R1 (mean reversion) or 1d trend turns up
            exit_signal = (close[i] > r1_aligned[i]) or (close[i] > ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dATR_Trend_VolumeRegime"
timeframe = "4h"
leverage = 1.0