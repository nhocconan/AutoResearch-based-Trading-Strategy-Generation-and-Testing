#!/usr/bin/env python3
"""
6h_Pivot_R1S1_Breakout_Volume_Adaptive
Hypothesis: Use 1D Camarilla R1/S1 with 6h entry, adaptive volume threshold (1.5x-2.5x based on volatility regime) and 1w trend filter to reduce whipsaw in sideways markets.
Long when price breaks above daily R1 with volume > (1.5 + 0.5 * vol_regime) * avg_volume during 08-20 UTC, only if weekly close > weekly open (bullish bias).
Short when price breaks below daily S1 with volume > (1.5 + 0.5 * vol_regime) * avg_volume during 08-20 UTC, only if weekly close < weekly open (bearish bias).
Volatility regime: low if ATR(14) < 50th percentile of ATR(50), high otherwise.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
Works in bull/bear via adaptive volume threshold and weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high - prev_low
    r1 = prev_close + range_1d * 1.1 / 12
    s1 = prev_close - range_1d * 1.1 / 12
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    weekly_bullish = close_1w > open_1w  # True for bullish week
    
    # Volatility regime: ATR-based
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(np.roll(close, 1) - low)
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    # Volatility regime: 0 = low volatility (ATR < 50th percentile), 1 = high volatility
    atr_50th_percentile = pd.Series(atr_50).rolling(window=100, min_periods=100).quantile(0.5).values
    vol_regime = np.where(atr_14 < atr_50th_percentile, 0.0, 1.0)  # 0=low vol, 1=high vol
    
    # Align all data to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need enough for ATR and alignments
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: adaptive threshold based on volatility regime
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma[i]):
            vol_confirm = False
        else:
            # Adaptive multiplier: 1.5 in low vol, 2.5 in high vol
            vol_multiplier = 1.5 + vol_regime_aligned[i]  # ranges from 1.5 to 2.5
            vol_confirm = volume[i] > vol_multiplier * vol_ma[i]
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and weekly bullish bias during session
            if (close[i] > r1_aligned[i] and vol_confirm and 
                weekly_bullish_aligned[i] > 0.5 and in_session):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and weekly bearish bias during session
            elif (close[i] < s1_aligned[i] and vol_confirm and 
                  weekly_bullish_aligned[i] < 0.5 and in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R1 or weekly bias turns bearish or outside session
            if (close[i] < r1_aligned[i] or 
                weekly_bullish_aligned[i] < 0.5 or 
                not in_session):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 or weekly bias turns bullish or outside session
            if (close[i] > s1_aligned[i] or 
                weekly_bullish_aligned[i] > 0.5 or 
                not in_session):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1S1_Breakout_Volume_Adaptive"
timeframe = "6h"
leverage = 1.0