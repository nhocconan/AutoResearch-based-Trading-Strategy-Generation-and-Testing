#!/usr/bin/env python3
"""
12h ATR Breakout with 1d Trend Filter and Volume Confirmation
Long when price breaks above ATR(14) upper band and 1d EMA50 > EMA200 (bullish trend)
Short when price breaks below ATR(14) lower band and 1d EMA50 < EMA200 (bearish trend)
Exit when price crosses back through ATR midpoint
Designed for low-frequency, high-conviction trades in trending markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_atr_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === ATR(14) Calculation ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0  # First value has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR Bands (using ATR multiplier of 1.5)
    atr_mult = 1.5
    atr_upper = close + atr_mult * atr
    atr_lower = close - atr_mult * atr
    atr_mid = (atr_upper + atr_lower) / 2
    
    # === Volume Filter (Volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # === 1d EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(atr[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below ATR midpoint
            if close[i] < atr_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above ATR midpoint
            if close[i] > atr_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Check volume filter first
            if not vol_filter[i]:
                signals[i] = 0.0
                continue
                
            # Bullish trend: EMA50 > EMA200
            if ema_50_aligned[i] > ema_200_aligned[i]:
                # Look for long entry: price breaks above ATR upper band
                if close[i] > atr_upper[i]:
                    position = 1
                    signals[i] = 0.25
            # Bearish trend: EMA50 < EMA200
            elif ema_50_aligned[i] < ema_200_aligned[i]:
                # Look for short entry: price breaks below ATR lower band
                if close[i] < atr_lower[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals