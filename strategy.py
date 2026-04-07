#!/usr/bin/env python3
"""
1h Trend Following with 4h/1d ADX Trend Filter and Volume Confirmation
- Uses 4h ADX > 25 and 1d price > SMA200 for trend direction (long only)
- 1h EMA(21) pullback to EMA(50) for entry timing
- Volume > 1.5x 20-period average for confirmation
- Fixed position size 0.20 to manage risk
- Designed for 15-30 trades/year to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_trend_adx_volume_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators ===
    # EMA21 and EMA50 for pullback entries
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 4h ADX for trend strength ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align indices
    
    # Directional Movement
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    def wilder_smooth(x, period):
        result = np.full_like(x, np.nan, dtype=float)
        if len(x) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(x[1:period])  # skip index 0
        for i in range(period, len(x)):
            result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    atr_4h = wilder_smooth(tr, 14)
    plus_di_4h = 100 * wilder_smooth(plus_dm, 14) / (atr_4h + 1e-10)
    minus_di_4h = 100 * wilder_smooth(minus_dm, 14) / (atr_4h + 1e-10)
    dx_4h = 100 * np.abs(plus_di_4h - minus_di_4h) / (plus_di_4h + minus_di_4h + 1e-10)
    adx_4h = wilder_smooth(dx_4h, 14)
    
    # Align 4h ADX to 1h
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # === 1d Trend Filter (price > SMA200) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0=flat, 1=long
    
    # Start after warmup period
    start_idx = max(50, 200)  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
            
        # Check for NaN values
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(sma200_1d_aligned[i]) or 
            np.isnan(ema21[i]) or np.isnan(ema50[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions: 4h ADX > 25 AND 1d price > SMA200
        trend_up = adx_4h_aligned[i] > 25 and close[i] > sma200_1d_aligned[i]
        
        if position == 1:
            # Exit: trend breaks down OR EMA21 crosses below EMA50
            if not trend_up or ema21[i] < ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        else:
            # Enter: trend up AND pullback to EMA21 (price near EMA21) AND volume confirmation
            pullback = abs(close[i] - ema21[i]) / ema21[i] < 0.01  # within 1% of EMA21
            if trend_up and pullback and vol_ratio[i] > 1.5:
                position = 1
                signals[i] = 0.20
            else:
                signals[i] = 0.0
    
    return signals