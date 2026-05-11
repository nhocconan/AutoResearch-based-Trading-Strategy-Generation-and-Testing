#!/usr/bin/env python3
"""
1h_4h1d_VWAP_Reversion_TrendFilter
Hypothesis: 1h price reverts to 4h VWAP (institutional fair value) with 1d EMA50 trend filter.
Long when: price < 4h VWAP - 0.5*ATR(14) AND 1d EMA50 rising AND volume > 1.5x 20-bar avg.
Short when: price > 4h VWAP + 0.5*ATR(14) AND 1d EMA50 falling AND volume > 1.5x 20-bar avg.
Exit when price crosses 4h VWAP or 1d EMA50 trend reverses.
Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.
Works in bull by buying dips to VWAP in uptrend; works in bear by selling rallies to VWAP in downtrend.
"""
name = "1h_4h1d_VWAP_Reversion_TrendFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for VWAP and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4h VWAP (typical price * volume cumulative) ---
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    vp_4h = typical_price_4h * df_4h['volume']
    cum_vp_4h = np.cumsum(vp_4h)
    cum_vol_4h = np.cumsum(df_4h['volume'])
    vwap_4h = np.where(cum_vol_4h != 0, cum_vp_4h / cum_vol_4h, np.nan)
    
    # --- 1d EMA50 trend ---
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i < 50:
            ema_1d[i] = np.nan
        elif i == 50:
            ema_1d[i] = np.mean(close_1d[0:50])
        else:
            ema_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_1d[i-1] * (49 / (50 + 1)))
    
    # EMA slope
    ema_slope_1d = np.full(len(close_1d), np.nan)
    for i in range(51, len(close_1d)):
        ema_slope_1d[i] = ema_1d[i] - ema_1d[i-1]
    
    # --- 1h ATR(14) for volatility scaling ---
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[0:14])
        else:
            atr[i] = (tr[i] * 1 / 14) + (atr[i-1] * 13 / 14)
    
    # --- 1h volume MA(20) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 4h VWAP and 1d indicators to 1h
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(4h VWAP needs 1 bar, EMA50, ATR14, vol MA20)
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap_4h_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # VWAP reversion conditions with volatility band
        vwap_upper = vwap_4h_aligned[i] + 0.5 * atr[i]
        vwap_lower = vwap_4h_aligned[i] - 0.5 * atr[i]
        
        price_below_vwap = close[i] < vwap_lower
        price_above_vwap = close[i] > vwap_upper
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if price_below_vwap and ema_slope_1d_aligned[i] > 0 and vol_spike:
                # Long: pullback to VWAP support in uptrend
                signals[i] = 0.20
                position = 1
            elif price_above_vwap and ema_slope_1d_aligned[i] < 0 and vol_spike:
                # Short: rally to VWAP resistance in downtrend
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses VWAP OR EMA50 trend turns down
                if close[i] > vwap_4h_aligned[i] or ema_slope_1d_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price crosses VWAP OR EMA50 trend turns up
                if close[i] < vwap_4h_aligned[i] or ema_slope_1d_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals