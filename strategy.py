#!/usr/bin/env python3
# 4h_1h_VWAP_Reversion_TrendFilter
# Hypothesis: 4h price reverts to 1-hour VWAP (institutional fair value) with 1h EMA20 trend filter.
# Long when: price < 1h VWAP - 0.5*ATR(14) AND 1h EMA20 rising AND volume > 1.5x 20-bar avg.
# Short when: price > 1h VWAP + 0.5*ATR(14) AND 1h EMA20 falling AND volume > 1.5x 20-bar avg.
# Exit when price crosses 1h VWAP or 1h EMA20 trend reverses.
# VWAP acts as a mean-reversion anchor; EMA20 filters counter-trend moves in bear markets.
# Works in bull by buying dips to VWAP in uptrend; works in bear by selling rallies to VWAP in downtrend.
# Target: 20-40 trades/year (80-160 total over 4 years) to avoid fee drag.

name = "4h_1h_VWAP_Reversion_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1h data for VWAP and EMA20
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1h VWAP (typical price * volume cumulative) ---
    typical_price_1h = (df_1h['high'] + df_1h['low'] + df_1h['close']) / 3
    vp_1h = typical_price_1h * df_1h['volume']
    cum_vp_1h = np.cumsum(vp_1h)
    cum_vol_1h = np.cumsum(df_1h['volume'])
    vwap_1h = np.where(cum_vol_1h != 0, cum_vp_1h / cum_vol_1h, np.nan)
    
    # --- 1h EMA20 trend ---
    close_1h = df_1h['close'].values
    ema_1h = np.full(len(close_1h), np.nan)
    for i in range(len(close_1h)):
        if i < 20:
            ema_1h[i] = np.nan
        elif i == 20:
            ema_1h[i] = np.mean(close_1h[0:20])
        else:
            ema_1h[i] = (close_1h[i] * 2 / (20 + 1)) + (ema_1h[i-1] * (19 / (20 + 1)))
    
    # EMA slope
    ema_slope_1h = np.full(len(close_1h), np.nan)
    for i in range(21, len(close_1h)):
        ema_slope_1h[i] = ema_1h[i] - ema_1h[i-1]
    
    # --- 4h ATR(14) for volatility scaling ---
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
    
    # --- 4h volume MA(20) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1h indicators to 4h
    vwap_1h_aligned = align_htf_to_ltf(prices, df_1h, vwap_1h)
    ema_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_1h)
    ema_slope_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_slope_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(1h VWAP needs 1 bar, EMA20, ATR14, vol MA20)
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap_1h_aligned[i]) or
            np.isnan(ema_1h_aligned[i]) or
            np.isnan(ema_slope_1h_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # VWAP reversion conditions with volatility band
        vwap_upper = vwap_1h_aligned[i] + 0.5 * atr[i]
        vwap_lower = vwap_1h_aligned[i] - 0.5 * atr[i]
        
        price_below_vwap = close[i] < vwap_lower
        price_above_vwap = close[i] > vwap_upper
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if price_below_vwap and ema_slope_1h_aligned[i] > 0 and vol_spike:
                # Long: pullback to VWAP support in uptrend
                signals[i] = 0.25
                position = 1
            elif price_above_vwap and ema_slope_1h_aligned[i] < 0 and vol_spike:
                # Short: rally to VWAP resistance in downtrend
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses VWAP OR EMA20 trend turns down
                if close[i] > vwap_1h_aligned[i] or ema_slope_1h_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses VWAP OR EMA20 trend turns up
                if close[i] < vwap_1h_aligned[i] or ema_slope_1h_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals