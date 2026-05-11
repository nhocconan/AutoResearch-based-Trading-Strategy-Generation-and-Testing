#!/usr/bin/env python3
"""
6h_Adaptive_Kelly_TRIX_Momentum_12hTrend
Hypothesis: Combine TRIX momentum with 12h trend filter and adaptive Kelly sizing based on TRIX signal strength. Works in bull/bear by aligning with higher timeframe trend while using TRIX zero-cross for entry timing. Adaptive sizing reduces exposure during weak signals, lowering drawdown. Target: 15-30 trades/year on 6h.
"""

name = "6h_Adaptive_Kelly_TRIX_Momentum_12hTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 12h Trend Filter (EMA34) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    ema34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # === TRIX Calculation (15,9,9) ===
    # TRIX = EMA(EMA(EMA(close,15),9),9) - 1 period ago, then % change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix_raw = ema3.pct_change() * 100  # percentage
    trix = trix_raw.values
    
    # === TRIX Signal Line (9-period EMA of TRIX) ===
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # === Volume Filter (1.5x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Signal Parameters ===
    base_size = 0.25  # base position size
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers TRIX and EMA calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line with uptrend (close > EMA34_12h) and volume
            if (trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1] and
                close[i] > ema34_12h_aligned[i] and volume_ok[i]):
                # Adaptive sizing based on TRIX strength (normalized)
                trix_strength = min(abs(trix[i]) / 0.5, 1.0)  # cap at 1.0 for 0.5 TRIX
                size = base_size * (0.5 + 0.5 * trix_strength)  # 0.5-1.0x base
                signals[i] = size
                position = 1
            # Short: TRIX crosses below signal line with downtrend (close < EMA34_12h) and volume
            elif (trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1] and
                  close[i] < ema34_12h_aligned[i] and volume_ok[i]):
                trix_strength = min(abs(trix[i]) / 0.5, 1.0)
                size = base_size * (0.5 + 0.5 * trix_strength)
                signals[i] = -size
                position = -1
        else:
            # Exit: TRIX crosses back through signal line
            if position == 1:
                if trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = base_size  # maintain base size while in trend
            elif position == -1:
                if trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -base_size
    
    return signals