#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volatility filter.
- Long when price breaks above Donchian upper band AND 1d close > 1d EMA50 AND ATR(14) < 1.5 * ATR(50)
- Short when price breaks below Donchian lower band AND 1d close < 1d EMA50 AND ATR(14) < 1.5 * ATR(50)
- Exit on opposite Donchian breakout (lower band for long exit, upper band for short exit)
- Uses 4h primary with 1d HTF to target 75-200 trades over 4 years (19-50/year)
- Donchian provides adaptive structure; EMA50 filters regime; ATR filter avoids high-volatility fakeouts
- Designed to work in both bull (breakouts with trend) and bear (mean reversion avoided via volatility filter)
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
    
    # Donchian(20) channels
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR(14) and ATR(50) for volatility filter
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe (waits for completed 1d bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_1d_aligned
    bearish_regime = close < ema_50_1d_aligned
    
    # Volatility filter: ATR(14) < 1.5 * ATR(50) (avoid high volatility chop)
    vol_filter = atr14 < (1.5 * atr50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50)  # Need Donchian and ATR50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper band AND bullish regime AND low volatility
            if close[i] > upper[i] and bullish_regime[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band AND bearish regime AND low volatility
            elif close[i] < lower[i] and bearish_regime[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below lower band (opposite Donchian level)
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above upper band (opposite Donchian level)
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_ATRVolFilter_v1"
timeframe = "4h"
leverage = 1.0