#!/usr/bin/env python3
"""
1d_Price_Channel_Breakout_v1
1d channel breakout with volume and ADX filter for trend strength.
Exit when price reverses through 10-day SMA or volatility filter.
Designed for low-frequency, high-conviction trades with minimal churn.
Target: 30-80 total trades over 4 years (7-20/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 20-day high/low for breakout ===
    high20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === ADX(14) for trend strength ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Volume average for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 10-day SMA for exit signal ===
    sma10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # === 1-week EMA10 for higher timeframe trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema10_1w = pd.Series(df_1w['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(high20[i]) or 
            np.isnan(low20[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(sma10[i]) or 
            np.isnan(ema10_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above 20-day high, ADX > 25, volume confirmed, price above 1w EMA10
            if (close[i] > high20[i] and 
                adx[i] > 25 and 
                vol_confirmed and 
                close[i] > ema10_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below 20-day low, ADX > 25, volume confirmed, price below 1w EMA10
            elif (close[i] < low20[i] and 
                  adx[i] > 25 and 
                  vol_confirmed and 
                  close[i] < ema10_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: trend weakening or reversal
        elif position == 1:
            # Exit long: ADX < 20 OR price crosses below SMA10
            if (adx[i] < 20 or 
                close[i] < sma10[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: ADX < 20 OR price crosses above SMA10
            if (adx[i] < 20 or 
                close[i] > sma10[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Price_Channel_Breakout_v1"
timeframe = "1d"
leverage = 1.0