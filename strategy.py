#!/usr/bin/env python3
# 12h_1d_volatility_breakout_v1
# Hypothesis: On 12h timeframe, price tends to break out of daily volatility bands (ATR-based) with strong momentum.
# Uses daily ATR multiplier to create dynamic bands. Enters long when price breaks above upper band with volume confirmation.
# Enters short when price breaks below lower band with volume confirmation.
# Exits when price returns to the daily close (mean reversion) or when volatility collapses.
# Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drift.
# Works in both bull and bear markets by capturing volatility expansion moves.

name = "12h_1d_volatility_breakout_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and reference levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(20) for volatility bands
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original length
    
    # ATR(20)
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Dynamic bands: ±1.5 * ATR from daily close
    upper_band = close_1d + 1.5 * atr_1d
    lower_band = close_1d - 1.5 * atr_1d
    
    # Align bands to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average (12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(daily_close_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above upper band
        if close[i] > upper_band_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below lower band
        elif close[i] < lower_band_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to daily close (mean reversion)
        elif position == 1 and close[i] <= daily_close_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= daily_close_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals