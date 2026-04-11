#!/usr/bin/env python3
# 4h_1d_4week_donchian_breakout_v1
# Strategy: 4h Donchian(20) breakout with 1d trend filter and 4-week volatility regime filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture momentum with clear structure. Combined with 1d EMA trend filter to avoid counter-trend trades and 4-week ATR percentile to filter low-volatility environments, this strategy works in both bull and bear markets by focusing on high-probability breakouts with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_4week_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load weekly data for 4-week ATR calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 4:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range for weekly
        tr1 = high_1w - low_1w
        tr2 = np.abs(high_1w - np.roll(close_1w, 1))
        tr3 = np.abs(low_1w - np.roll(close_1w, 1))
        tr1[0] = 0  # First value has no previous close
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_4w = pd.Series(tr).rolling(window=4, min_periods=4).mean().values
        
        # Percentile rank of current ATR over 52 weeks (1 year)
        atr_percentile = pd.Series(atr_4w).rolling(window=52, min_periods=10).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
        ).values
        atr_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile, additional_delay_bars=0)
    else:
        # Fallback if insufficient weekly data
        atr_percentile_aligned = np.full(n, 0.5)
    
    # Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = pd.Series(volume) / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(vol_ratio.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: only trade when ATR percentile > 0.3 (avoid low volatility)
        vol_filter = atr_percentile_aligned[i] > 0.3
        
        # Volume confirmation
        vol_confirmed = vol_ratio.iloc[i] > 1.3
        
        # Entry conditions
        # Long: Price breaks above Donchian upper + uptrend (price > 1d EMA50) + vol filter + volume
        if (vol_filter and vol_confirmed and 
            close[i] > highest_20[i] and 
            close[i] > ema_50_1d_aligned[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short: Price breaks below Donchian lower + downtrend (price < 1d EMA50) + vol filter + volume
        elif (vol_filter and vol_confirmed and 
              close[i] < lowest_20[i] and 
              close[i] < ema_50_1d_aligned[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: price reverts to middle of Donchian channel or trend reversal
        elif position == 1 and (close[i] < (highest_20[i] + lowest_20[i]) / 2 or close[i] < ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > (highest_20[i] + lowest_20[i]) / 2 or close[i] > ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals