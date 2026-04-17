#!/usr/bin/env python3
"""
4h_Keltner_Breakout_Channel_v1
Breakout above/below Keltner Channel (20, 2.0) with ADX(14) > 20 trend filter.
Exit when price returns to middle line (EMA20).
Uses 12h EMA50 for higher timeframe trend alignment.
Designed to capture sustained moves with volatility-adjusted bands.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === EMA20 (middle line) ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === ATR(10) for Keltner width ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # === Keltner Bands ===
    keltner_upper = ema20 + 2.0 * atr
    keltner_lower = ema20 - 2.0 * atr
    
    # === ADX(14) for trend strength ===
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === 12h EMA50 for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20[i]) or 
            np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper band, ADX > 20, price above 12h EMA50
            if (close[i] > keltner_upper[i] and 
                adx[i] > 20 and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower band, ADX > 20, price below 12h EMA50
            elif (close[i] < keltner_lower[i] and 
                  adx[i] > 20 and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to middle line (EMA20)
        elif position == 1:
            # Exit long: price crosses below EMA20
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA20
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Keltner_Breakout_Channel_v1"
timeframe = "4h"
leverage = 1.0