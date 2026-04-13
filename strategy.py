#!/usr/bin/env python3
"""
4h_1d_Kelly_Criterion_With_Volatility_Scaling
Hypothesis: Kelly criterion-based position sizing with volatility scaling adapts to market conditions,
providing optimal risk-adjusted returns. Uses daily volatility (ATR) to scale position size and
4h EMA crossover for trend direction. Volatility scaling reduces size in high volatility (crisis)
periods and increases in low volatility, improving Sharpe ratio. Targets 20-40 trades/year.
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
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) on daily
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 4h data for EMA crossover
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily ATR and 4h EMAs to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volatility-adjusted Kelly scaling: base Kelly fraction scaled by inverse volatility
    # Use 20-day average ATR as reference for normalization
    atr_ma_20 = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean()
    # Avoid division by zero and extreme values
    vol_ratio = np.where(atr_ma_20 > 0, atr_ma_20.iloc if hasattr(atr_ma_20, 'iloc') else atr_ma_20, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    # Invert volatility: higher vol -> lower position size
    vol_scaling = np.where(vol_ratio > 0, 1.0 / vol_ratio, 1.0)
    vol_scaling = np.clip(vol_scaling, 0.5, 2.0)  # Limit scaling to reasonable range
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    base_size = 0.25  # Base position size before volatility scaling
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_scaling[i])):
            signals[i] = 0.0
            continue
        
        # Calculate volatility-adjusted position size
        vol_adj_size = base_size * vol_scaling[i]
        vol_adj_size = min(vol_adj_size, 0.40)  # Enforce max position size
        
        # EMA crossover signals
        ema_bullish = ema_20_aligned[i] > ema_50_aligned[i]
        ema_bearish = ema_20_aligned[i] < ema_50_aligned[i]
        
        if ema_bullish and position != 1:
            position = 1
            signals[i] = vol_adj_size
        elif ema_bearish and position != -1:
            position = -1
            signals[i] = -vol_adj_size
        else:
            # Hold current position with volatility-adjusted size
            if position == 1:
                signals[i] = vol_adj_size
            elif position == -1:
                signals[i] = -vol_adj_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_Kelly_Criterion_With_Volatility_Scaling"
timeframe = "4h"
leverage = 1.0