# 4h_1d_VWAP_Reversion_TrendFilter
# Hypothesis: 4h price reverts to 1-day VWAP (institutional fair value) with trend filter.
# Long when: price < 1d VWAP - 0.5*ATR(14) AND 1d EMA20 rising AND volume > 1.5x 20-bar avg.
# Short when: price > 1d VWAP + 0.5*ATR(14) AND 1d EMA20 falling AND volume > 1.5x 20-bar avg.
# Exit when price crosses 1d VWAP or 1d EMA20 trend reverses.
# VWAP acts as a mean-reversion anchor; EMA20 filters counter-trend moves in bear markets.
# Works in bull by buying dips to VWAP in uptrend; works in bore by selling rallies to VWAP in downtrend.
# Target: 20-40 trades/year (80-160 total over 4 years) to avoid fee drag.

name = "4h_1d_VWAP_Reversion_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for VWAP and EMA20
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d VWAP (typical price * volume cumulative) ---
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vp_1d = typical_price_1d * df_1d['volume']
    cum_vp_1d = np.cumsum(vp_1d)
    cum_vol_1d = np.cumsum(df_1d['volume'])
    vwap_1d = np.where(cum_vol_1d != 0, cum_vp_1d / cum_vol_1d, np.nan)
    
    # --- 1d EMA20 trend ---
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i < 20:
            ema_1d[i] = np.nan
        elif i == 20:
            ema_1d[i] = np.mean(close_1d[0:20])
        else:
            ema_1d[i] = (close_1d[i] * 2 / (20 + 1)) + (ema_1d[i-1] * (19 / (20 + 1)))
    
    # EMA slope
    ema_slope_1d = np.full(len(close_1d), np.nan)
    for i in range(21, len(close_1d)):
        ema_slope_1d[i] = ema_1d[i] - ema_1d[i-1]
    
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
    
    # Align 1d indicators to 4h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(1d VWAP needs 1 bar, EMA20, ATR14, vol MA20)
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap_1d_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # VWAP reversion conditions with volatility band
        vwap_upper = vwap_1d_aligned[i] + 0.5 * atr[i]
        vwap_lower = vwap_1d_aligned[i] - 0.5 * atr[i]
        
        price_below_vwap = close[i] < vwap_lower
        price_above_vwap = close[i] > vwap_upper
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if price_below_vwap and ema_slope_1d_aligned[i] > 0 and vol_spike:
                # Long: pullback to VWAP support in uptrend
                signals[i] = 0.25
                position = 1
            elif price_above_vwap and ema_slope_1d_aligned[i] < 0 and vol_spike:
                # Short: rally to VWAP resistance in downtrend
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses VWAP OR EMA20 trend turns down
                if close[i] > vwap_1d_aligned[i] or ema_slope_1d_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses VWAP OR EMA20 trend turns up
                if close[i] < vwap_1d_aligned[i] or ema_slope_1d_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals