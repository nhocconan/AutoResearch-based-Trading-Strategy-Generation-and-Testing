#!/usr/bin/env python3
# 1d_VWAP_Reversion_TrendFilter
# Hypothesis: 1d price reverts to weekly VWAP with weekly trend filter. Long when price < weekly VWAP - 0.5*ATR(14) AND weekly EMA20 rising AND volume > 1.5x 20-bar avg. Short when price > weekly VWAP + 0.5*ATR(14) AND weekly EMA20 falling AND volume > 1.5x 20-bar avg. Exit when price crosses weekly VWAP or weekly EMA20 trend reverses. VWAP acts as a mean-reversion anchor; EMA20 filters counter-trend moves in bear markets. Works in bull by buying dips to VWAP in uptrend; works in bear by selling rallies to VWAP in downtrend. Target: 10-25 trades/year (40-100 total over 4 years) to avoid fee drag.

name = "1d_VWAP_Reversion_TrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for VWAP and EMA20
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1w VWAP (typical price * volume cumulative) ---
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vp_1w = typical_price_1w * df_1w['volume']
    cum_vp_1w = np.cumsum(vp_1w)
    cum_vol_1w = np.cumsum(df_1w['volume'])
    vwap_1w = np.where(cum_vol_1w != 0, cum_vp_1w / cum_vol_1w, np.nan)
    
    # --- 1w EMA20 trend ---
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        if i < 20:
            ema_1w[i] = np.nan
        elif i == 20:
            ema_1w[i] = np.mean(close_1w[0:20])
        else:
            ema_1w[i] = (close_1w[i] * 2 / (20 + 1)) + (ema_1w[i-1] * (19 / (20 + 1)))
    
    # EMA slope
    ema_slope_1w = np.full(len(close_1w), np.nan)
    for i in range(21, len(close_1w)):
        ema_slope_1w[i] = ema_1w[i] - ema_1w[i-1]
    
    # --- 1d ATR(14) for volatility scaling ---
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
    
    # --- 1d volume MA(20) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1w indicators to 1d
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    ema_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(1w VWAP needs 1 bar, EMA20, ATR14, vol MA20)
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap_1w_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(ema_slope_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # VWAP reversion conditions with volatility band
        vwap_upper = vwap_1w_aligned[i] + 0.5 * atr[i]
        vwap_lower = vwap_1w_aligned[i] - 0.5 * atr[i]
        
        price_below_vwap = close[i] < vwap_lower
        price_above_vwap = close[i] > vwap_upper
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if price_below_vwap and ema_slope_1w_aligned[i] > 0 and vol_spike:
                # Long: pullback to VWAP support in uptrend
                signals[i] = 0.25
                position = 1
            elif price_above_vwap and ema_slope_1w_aligned[i] < 0 and vol_spike:
                # Short: rally to VWAP resistance in downtrend
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses VWAP OR EMA20 trend turns down
                if close[i] > vwap_1w_aligned[i] or ema_slope_1w_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses VWAP OR EMA20 trend turns up
                if close[i] < vwap_1w_aligned[i] or ema_slope_1w_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals