#!/usr/bin/env python3
"""
Hypothesis: 6h ADX + MACD trend filter with 12h/1d confluence.
Uses 12h ADX > 25 to confirm trend strength, 1d MACD histogram for entry timing,
and 6h EMA20 as dynamic support/resistance. Filters avoid whipsaws in ranging markets.
Trades only when higher timeframe trend is strong and aligned.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_macd_12h1d_confluence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h ADX for trend strength (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate ADX on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 1d MACD for entry timing (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # MACD: 12, 26, 9
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close_1d).ewm(span=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    macd_hist = macd_line - signal_line
    
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist)
    
    # === 6h EMA20 for dynamic support/resistance (LTF) ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        if np.isnan(adx_aligned[i]) or np.isnan(macd_hist_aligned[i]) or np.isnan(ema20[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: MACD histogram turns negative OR trend weakens
            if macd_hist_aligned[i] < 0 or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: MACD histogram turns positive OR trend weakens
            if macd_hist_aligned[i] > 0 or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need strong trend
            if not strong_trend:
                signals[i] = 0.0
                continue
            
            # Entry: MACD histogram crossover with price relative to EMA20
            if macd_hist_aligned[i] > 0 and close[i] > ema20[i]:
                # Bullish momentum above EMA20 -> long
                position = 1
                signals[i] = 0.25
            elif macd_hist_aligned[i] < 0 and close[i] < ema20[i]:
                # Bearish momentum below EMA20 -> short
                position = -1
                signals[i] = -0.25
    
    return signals