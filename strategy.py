#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_Filter
Hypothesis: 4-hour Donchian(20) breakout in the direction of the 1-day EMA50 trend, confirmed by volume expansion, works in both bull and bear markets by capturing momentum bursts. Uses 1-day trend filter to avoid counter-trend trades, reducing whipsaw in ranging/ bearish conditions. Targets 20-50 trades/year.
"""

name = "4h_Donchian20_Breakout_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d EMA50 for trend filter ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 1d ATR for volatility filter ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / (atr_ma_1d + 1e-10)
    atr_ratio_4h_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # --- 4h Donchian(20) channels ---
    highest_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # --- 4h Volume average for confirmation ---
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(highest_high_4h[i]) or 
            np.isnan(lowest_low_4h[i]) or np.isnan(atr_ratio_4h_aligned[i]) or 
            np.isnan(vol_avg_4h[i])):
            if position != 0:
                # Simple stoploss: 2.5x ATR from entry
                atr_est = np.abs(high_4h[i] - low_4h[i])  # rough 4h ATR estimate
                if position == 1 and close_4h[i] <= entry_price - 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close_4h[i] > ema50_1d_aligned[i]
        downtrend = close_4h[i] < ema50_1d_aligned[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_ratio_4h_aligned[i] > 0.5
        
        # Volume confirmation: current volume > 1.5x 4h average
        vol_confirm = volume_4h[i] > 1.5 * vol_avg_4h[i]
        
        if position == 0:
            # Look for breakout entries in direction of trend
            if vol_filter and vol_confirm:
                # Long breakout above upper Donchian in uptrend
                if uptrend and close_4h[i] > highest_high_4h[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_4h[i]
                # Short breakdown below lower Donchian in downtrend
                elif downtrend and close_4h[i] < lowest_low_4h[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_4h[i]
        else:
            # Manage existing position: exit on opposite Donchian breach or volatility collapse
            if position == 1:
                # Long: exit if price breaks below lower Donchian or volatility collapses
                if close_4h[i] < lowest_low_4h[i] or atr_ratio_4h_aligned[i] < 0.3:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short: exit if price breaks above upper Donchian or volatility collapses
                if close_4h[i] > highest_high_4h[i] or atr_ratio_4h_aligned[i] < 0.3:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals