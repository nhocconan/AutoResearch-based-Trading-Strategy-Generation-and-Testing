#!/usr/bin/env python3
"""
6h_AdaptiveMFE_MAE_Regime
Hypothesis: On 6h timeframe, enter long when price is above 20-period EMA, MFE (max favorable excursion) over last 3 bars exceeds MAE (max adverse excursion) by 1.5x, and 1d ATR ratio indicates low volatility regime (ATR(7)/ATR(30) < 0.8). Enter short when price below EMA, MAE exceeds MFE by 1.5x, and same low vol regime. Uses MFE/MAE to detect momentum persistence within low volatility regimes, which often precedes explosive moves. Designed for 50-150 total trades over 4 years to minimize fee drag and work in both bull/bear markets via volatility regime filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === 20-period EMA on 6h ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === Daily ATR for volatility regime ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr7 / atr30  # < 0.8 indicates low volatility regime
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # === MFE and MAE over last 3 bars ===
    mfe = np.zeros(n)  # max favorable excursion
    mae = np.zeros(n)  # max adverse excursion
    
    for i in range(2, n):
        if close[i] >= close[i-1]:  # up or equal
            # For long bias: favorable = upward movement, adverse = downward
            mfe[i] = max(
                high[i] - close[i-2],  # max favorable from 2 bars ago
                high[i-1] - close[i-2],
                0
            )
            mae[i] = max(
                close[i-2] - low[i],
                close[i-2] - low[i-1],
                0
            )
        else:  # down
            # For short bias: favorable = downward movement, adverse = upward
            mfe[i] = max(
                close[i-2] - low[i],
                close[i-2] - low[i-1],
                0
            )
            mae[i] = max(
                high[i] - close[i-2],
                high[i-1] - close[i-2],
                0
            )
    
    # Smooth MFE/MAE to reduce noise
    mfe_smooth = pd.Series(mfe).ewm(span=3, adjust=False, min_periods=3).mean().values
    mae_smooth = pd.Series(mae).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: covers EMA20, ATR, and MFE/MAE initialization
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema20[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(mfe_smooth[i]) or np.isnan(mae_smooth[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Low volatility regime filter
        low_vol_regime = atr_ratio_aligned[i] < 0.8
        
        # MFE/MAE ratio
        mfe_mae_ratio = mfe_smooth[i] / (mae_smooth[i] + 1e-10)  # avoid division by zero
        mae_mfe_ratio = mae_smooth[i] / (mfe_smooth[i] + 1e-10)
        
        # Entry conditions
        if position == 0 and low_vol_regime:
            # Long: price above EMA20, MFE > 1.5 * MAE
            if close[i] > ema20[i] and mfe_mae_ratio > 1.5:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below EMA20, MAE > 1.5 * MFE
            elif close[i] < ema20[i] and mae_mfe_ratio > 1.5:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse when price crosses EMA20
        elif position == 1:
            if close[i] < ema20[i]:  # price crosses below EMA20
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > ema20[i]:  # price crosses above EMA20
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_AdaptiveMFE_MAE_Regime"
timeframe = "6h"
leverage = 1.0