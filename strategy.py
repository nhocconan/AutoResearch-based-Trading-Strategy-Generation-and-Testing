#!/usr/bin/env python3
"""
4h_ADX_CCI_Breakout_v1
ADX(14) > 25 + CCI(20) > 100 for long, CCI(20) < -100 for short.
Uses 12h timeframe for trend filter: price above/below 12h EMA50.
Exit when CCI crosses back within [-50, 50] or ADX < 20.
Designed to capture strong trends with momentum confirmation.
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
    
    # === CCI(20) ===
    typical_price = (high + low + close) / 3.0
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci = (typical_price - sma_tp) / (0.015 * mad)
    
    # === ADX(14) ===
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
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
        if (np.isnan(cci[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: ADX > 25, CCI > 100, price above 12h EMA50
            if (adx[i] > 25 and 
                cci[i] > 100 and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: ADX > 25, CCI < -100, price below 12h EMA50
            elif (adx[i] > 25 and 
                  cci[i] < -100 and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: CCI < 50 OR ADX < 20
            if (cci[i] < 50 or 
                adx[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: CCI > -50 OR ADX < 20
            if (cci[i] > -50 or 
                adx[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ADX_CCI_Breakout_v1"
timeframe = "4h"
leverage = 1.0