# 4H_NADIR_Momentum_v1
# Nadir-weighted momentum with volatility regime filter.
# Uses momentum with lookback = 40 (10-day equivalent in 4h) and Nadir weighting.
# Nadir weighting gives exponentially more weight to recent momentum lows,
# making it sensitive to emerging momentum shifts while rejecting noise.
# Combined with 1D volatility regime filter (ATR ratio < 0.8 = low vol environment)
# to avoid whipsaw in choppy markets. Designed to work in both bull and bear
# by capturing momentum shifts regardless of direction, with volatility filter
# reducing false signals during high volatility periods.
# Target: 20-50 trades/year (~80-200 total over 4 years).

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
    
    # === Nadir-weighted momentum (40 period) ===
    # Momentum = close - close[40]
    momentum = close - np.roll(close, 40)
    momentum[:40] = 0  # First 40 values undefined
    
    # Nadir weights: exponential decay with emphasis on recent lows
    # Weight = exp(-5 * (1 - momentum/max_momentum)) for normalization
    # Actually simpler: use sigmoid of momentum rank to emphasize extremes
    momentum_abs = np.abs(momentum)
    momentum_rank = pd.Series(momentum_abs).rolling(window=80, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Nadir weighting: emphasize when momentum is at recent extremes (low rank = low momentum abs)
    nadir_weight = 1.0 - momentum_rank  # High weight when momentum is weak
    nadir_momentum = momentum * nadir_weight
    
    # Smooth the nadir momentum
    nadir_momentum_smooth = pd.Series(nadir_momentum).ewm(
        span=10, adjust=False, min_periods=10
    ).mean().values
    
    # === 1D volatility regime filter ===
    df_1d = get_htf_data(prices, '1d')
    atr_14_1d = pd.Series(
        np.maximum(
            np.maximum(df_1d['high'] - df_1d['low'],
                       np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))),
            np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
        )
    ).rolling(window=14, min_periods=14).mean().values
    
    atr_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14_1d / (atr_50_1d + 1e-10)  # Current ATR vs 50-period average
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volatility regime: low volatility environment (ATR ratio < 0.8)
    vol_regime = atr_ratio_aligned < 0.8
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(nadir_momentum_smooth[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: positive nadir momentum AND low volatility regime
            if (nadir_momentum_smooth[i] > 0 and 
                vol_regime[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: negative nadir momentum AND low volatility regime
            elif (nadir_momentum_smooth[i] < 0 and 
                  vol_regime[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or volatility regime change
        elif position == 1:
            # Exit long: negative momentum OR high volatility
            if (nadir_momentum_smooth[i] < 0 or 
                not vol_regime[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: positive momentum OR high volatility
            if (nadir_momentum_smooth[i] > 0 or 
                not vol_regime[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4H_NADIR_Momentum_v1"
timeframe = "4h"
leverage = 1.0