#!/usr/bin/env python3
# 4h_CCI_Reversal_Trend
# Hypothesis: Use 4-hour CCI for mean-reversion entries in range-bound markets,
# filtered by 1-day ADX to avoid strong trends and volume spikes for confirmation.
# Works in both bull and bear markets by focusing on mean-reversion in ranging
# regimes (ADX < 25) while avoiding trending conditions. Low-frequency design
# targets 20-40 trades/year to minimize fee drag.

name = "4h_CCI_Reversal_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_cci(high, low, close, period=20):
    """Calculate Commodity Channel Index (CCI)."""
    tp = (high + low + close) / 3.0
    sma = pd.Series(tp).rolling(window=period, min_periods=period).mean()
    mad = pd.Series(tp).rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
    )
    cci = (tp - sma) / (0.015 * mad)
    return cci.fillna(0).values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().fillna(0).values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h CCI for mean-reversion signals
    cci = calculate_cci(high, low, close, period=20)
    
    # Calculate daily ADX for trend strength filter
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, period=14)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(cci[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: only trade in ranging markets (ADX < 25)
        ranging_market = adx_1d_aligned[i] < 25
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # CCI reversal signals
        cci_oversold = cci[i] < -100  # Strong oversold
        cci_overbought = cci[i] > 100  # Strong overbought
        cci_exit = abs(cci[i]) < 50   # Return to neutral zone
        
        if position == 0:
            # LONG: CCI oversold + ranging market + volume confirmation
            if cci_oversold and ranging_market and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: CCI overbought + ranging market + volume confirmation
            elif cci_overbought and ranging_market and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: CCI returns to neutral or trend strengthens
            if cci_exit or not ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CCI returns to neutral or trend strengthens
            if cci_exit or not ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals