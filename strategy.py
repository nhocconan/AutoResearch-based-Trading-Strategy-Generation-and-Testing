#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 12h volume-weighted trend filter and adaptive ATR position sizing.
- Long when price breaks above Donchian(20) upper band AND 12h VWAP > 12h EMA50 (bullish regime)
- Short when price breaks below Donchian(20) lower band AND 12h VWAP < 12h EMA50 (bearish regime)
- Position size scaled by ATR volatility (0.20 in low vol, 0.30 in high vol) to manage drawdown
- Exit on opposite Donchian breakout or trend regime change (VWAP/EMA50 crossover)
- Uses 4h primary with 12h HTF to target 75-200 trades over 4 years (19-50/year)
- Donchian provides objective breakout levels; VWAP/EMA50 combo filters chop and confirms institutional flow
- Adaptive sizing reduces exposure during high volatility periods (e.g., 2022 crash) while maintaining upside
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR(14) for volatility-based position sizing
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Get 12h data ONCE before loop for regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h VWAP and EMA50
    vwap_12h = (df_12h['close'] * df_12h['volume']).expanding().sum() / df_12h['volume'].expanding().sum()
    vwap_12h = vwap_12h.values
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h timeframe
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Regime: bullish if VWAP > EMA50, bearish if VWAP < EMA50
    bullish_regime = vwap_12h_aligned > ema_50_12h_aligned
    bearish_regime = vwap_12h_aligned < ema_50_12h_aligned
    
    # ATR-based position sizing: normalize ATR to [0,1] over 50-period, scale size 0.20-0.30
    atr_ratio = pd.Series(atr).rolling(window=50, min_periods=10).apply(
        lambda x: (x[-1] - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-10), raw=False
    ).values
    atr_ratio = np.nan_to_num(atr_ratio, nan=0.5)
    position_size = 0.20 + 0.10 * atr_ratio  # 0.20 to 0.30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vwap_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(position_size[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper band AND bullish regime
            if close[i] > upper[i] and bullish_regime[i]:
                signals[i] = position_size[i]
                position = 1
            # Short: break below lower band AND bearish regime
            elif close[i] < lower[i] and bearish_regime[i]:
                signals[i] = -position_size[i]
                position = -1
        elif position == 1:
            # Long exit: break below lower band OR regime turns bearish
            if close[i] < lower[i] or bearish_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size[i]
        elif position == -1:
            # Short exit: break above upper band OR regime turns bullish
            if close[i] > upper[i] or bullish_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size[i]
    
    return signals

name = "4h_Donchian20_12hVWAP_EMA50_Regime_v1"
timeframe = "4h"
leverage = 1.0