#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h ADX for trend strength and 1d Donchian breakouts for entry.
# Goes long when 4h ADX > 25 (strong trend) and price breaks above 1d Donchian high.
# Goes short when 4h ADX > 25 and price breaks below 1d Donchian low.
# Uses 4h EMA for trend direction filter and ATR-based stops.
# Designed to capture strong trends while avoiding choppy markets.
# Target: 60-150 total trades over 4 years (15-37/year) with low frequency.

name = "1h_adx25_donchian1d_ema4h_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h data for ADX and EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    
    # ADX
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_4h = adx
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 4h EMA(20) for trend direction
    ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels (20-period) on 1d
    high_max_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    high_max_1d_aligned = align_htf_to_ltf(prices, df_1d, high_max_1d)
    low_min_1d_aligned = align_htf_to_ltf(prices, df_1d, low_min_1d)
    
    # ATR for stoploss (using 1h data)
    tr1_h = high - low
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr2_h[0] = 0
    tr3_h[0] = 0
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr = pd.Series(tr_h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(high_max_1d_aligned[i]) or np.isnan(low_min_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Trend strength filter: ADX > 25
        strong_trend = adx_4h_aligned[i] > 25
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend weakens or price breaks below 4h EMA
            elif not strong_trend or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend weakens or price breaks above 4h EMA
            elif not strong_trend or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with strong trend
            if strong_trend:
                # Long entry: price breaks above 1d Donchian high with 4h EMA uptrend
                if close[i] > high_max_1d_aligned[i] and close[i] > ema_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short entry: price breaks below 1d Donchian low with 4h EMA downtrend
                elif close[i] < low_min_1d_aligned[i] and close[i] < ema_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
    
    return signals