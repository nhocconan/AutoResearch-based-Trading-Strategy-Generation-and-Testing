#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day ATR-based volatility filter and price channel breakout
# - Long when price breaks above 12h Donchian(20) upper band with 1-day ATR contraction (volatility squeeze)
# - Short when price breaks below 12h Donchian(20) lower band with 1-day ATR contraction
# - Exit when price reverts to 12h Donchian(20) middle band
# - Volatility filter: only trade when 1-day ATR ratio (current/20-period MA) < 0.8 (low volatility environment)
# - Position sizing: 0.25 for controlled risk
# - Target: 50-150 total trades over 4 years (12-37/year) with low frequency to avoid fee drag

name = "12h_DonchianBreakout_VolatilitySqueeze"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = tr1[0]  # First period
    tr3[0] = tr1[0]  # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(20) on daily
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ma_20 = pd.Series(atr_20).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_20 / atr_ma_20  # Current ATR relative to its 20-day MA
    
    # Align 1d ATR ratio to 12h timeframe
    atr_ratio_12h = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 12h Donchian channels (20-period)
    # Using rolling window on 12h data directly
    dh_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dh_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    dh_mid = (dh_high + dh_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(dh_high[i]) or np.isnan(dh_low[i]) or 
            np.isnan(atr_ratio_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper band with low volatility (squeeze)
            if close[i] > dh_high[i] and atr_ratio_12h[i] < 0.8:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with low volatility (squeeze)
            elif close[i] < dh_low[i] and atr_ratio_12h[i] < 0.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to middle band
            if close[i] <= dh_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to middle band
            if close[i] >= dh_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals