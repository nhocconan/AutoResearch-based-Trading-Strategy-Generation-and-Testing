#!/usr/bin/env python3
"""
4h_Momentum_Breakout_With_Volume_Regime_v1
Momentum strategy combining 4h RSI(2) extreme readings with price breaking above/below 
4h Donchian(20) channels, confirmed by volume spike and filtered by 1d ADX(14) regime.
Exit when RSI returns to neutral zone (40-60) or price reverses through 4h EMA(10).
Designed to capture momentum bursts in both trending and ranging markets with controlled risk.
Target: 80-150 total trades over 4 years (20-38/year).
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
    volume = prices['volume'].values
    
    # === RSI(2) for momentum extreme ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Donchian(20) channels for breakout ===
    high20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === EMA(10) for exit signal ===
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d ADX(14) for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(high20[i]) or 
            np.isnan(low20[i]) or 
            np.isnan(ema10[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        # Regime filter: only trade when 1d ADX > 20 (trending market)
        regime_filter = adx_14_aligned[i] > 20
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI < 10 (extremely oversold) AND price breaks above Donchian high 
            #         AND volume confirmed AND regime filter
            if (rsi[i] < 10 and 
                close[i] > high20[i] and 
                vol_confirmed and 
                regime_filter):
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI > 90 (extremely overbought) AND price breaks below Donchian low
            #          AND volume confirmed AND regime filter
            elif (rsi[i] > 90 and 
                  close[i] < low20[i] and 
                  vol_confirmed and 
                  regime_filter):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: RSI returns to neutral zone OR price reverses through EMA10
        elif position == 1:
            # Exit long: RSI > 40 (returning from oversold) OR price crosses below EMA10
            if (rsi[i] > 40 or 
                close[i] < ema10[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 60 (returning from overbought) OR price crosses above EMA10
            if (rsi[i] < 60 or 
                close[i] > ema10[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Momentum_Breakout_With_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0