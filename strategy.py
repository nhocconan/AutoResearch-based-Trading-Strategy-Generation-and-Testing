#!/usr/bin/env python3
"""
6h Volatility Breakout + Trend Confirmation
Hypothesis: Combines volatility expansion (ATR-based breakout) with trend confirmation (EMA alignment) to capture strong moves in both bull and bear markets. Uses 1d EMA for higher timeframe trend context. Volatility filter ensures trades only during active markets. Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14379_6h_vol_breakout_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA trend (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA21 for short-term trend
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.7 * vol_ma)  # Require at least 70% of average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(200, 21, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(ema21[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR EMA21 turns bearish OR stoploss
            if (close[i] <= donchian_low[i] or ema21[i] < ema200_1d_aligned[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR EMA21 turns bullish OR stoploss
            if (close[i] >= donchian_high[i] or ema21[i] > ema200_1d_aligned[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: volatility breakout + trend alignment + volume
            # Long: price breaks above Donchian high + above EMA200 trend + volume
            long_breakout = close[i] > donchian_high[i]
            long_trend = ema21[i] > ema200_1d_aligned[i]  # EMA21 above 1d EMA200
            long_setup = long_breakout and long_trend and vol_filter[i]
            
            # Short: price breaks below Donchian low + below EMA200 trend + volume
            short_breakout = close[i] < donchian_low[i]
            short_trend = ema21[i] < ema200_1d_aligned[i]  # EMA21 below 1d EMA200
            short_setup = short_breakout and short_trend and vol_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals