#!/usr/bin/env python3
# 4h_1w_SuperTrend_Filter
# Hypothesis: Use 1-week SuperTrend as primary trend filter and 4h price action for entries.
# Long when: 4h close crosses above 4h EMA20 AND 1w SuperTrend is bullish AND volume > 1.5x 20-bar avg.
# Short when: 4h close crosses below 4h EMA20 AND 1w SuperTrend is bearish AND volume > 1.5x 20-bar avg.
# Exit when 4h close crosses back over EMA20 or SuperTrend flips.
# SuperTrend on weekly timeframe filters out noise and avoids counter-trend trades in choppy markets.
# Works in bull by catching pullbacks to EMA20 in uptrend; works in bear by selling rallies to EMA20 in downtrend.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "4h_1w_SuperTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for SuperTrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4h EMA20 ---
    close_4h = close
    ema_20 = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            ema_20[i] = np.nan
        elif i == 20:
            ema_20[i] = np.mean(close_4h[0:20])
        else:
            ema_20[i] = (close_4h[i] * 2 / (20 + 1)) + (ema_20[i-1] * (19 / (20 + 1)))
    
    # --- 1w SuperTrend (ATR=10, multiplier=3) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    
    # ATR(10)
    atr_1w = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        if i < 10:
            atr_1w[i] = np.nan
        elif i == 10:
            atr_1w[i] = np.mean(tr_1w[0:10])
        else:
            atr_1w[i] = (tr_1w[i] * 2 / (10 + 1)) + (atr_1w[i-1] * (9 / (10 + 1)))
    
    # SuperTrend calculation
    upper_band = np.full(len(close_1w), np.nan)
    lower_band = np.full(len(close_1w), np.nan)
    supertrend = np.full(len(close_1w), np.nan)
    trend = np.full(len(close_1w), np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, len(close_1w)):
        if np.isnan(atr_1w[i]):
            continue
        upper_band[i] = ((high_1w[i] + low_1w[i]) / 2) + 3 * atr_1w[i]
        lower_band[i] = ((high_1w[i] + low_1w[i]) / 2) - 3 * atr_1w[i]
        
        if i == 10:
            supertrend[i] = upper_band[i]
            trend[i] = -1  # start in downtrend
        else:
            if supertrend[i-1] == upper_band[i-1]:
                if close_1w[i] <= upper_band[i]:
                    supertrend[i] = upper_band[i]
                    trend[i] = -1
                else:
                    supertrend[i] = lower_band[i]
                    trend[i] = 1
            else:
                if close_1w[i] >= lower_band[i]:
                    supertrend[i] = lower_band[i]
                    trend[i] = 1
                else:
                    supertrend[i] = upper_band[i]
                    trend[i] = -1
    
    # --- 4h volume MA(20) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1w indicators to 4h
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_1w, trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(EMA20, SuperTrend, vol MA)
    start_idx = max(20, 10, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20[i]) or
            np.isnan(supertrend_aligned[i]) or
            np.isnan(trend_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price relative to EMA20
        price_above_ema = close[i] > ema_20[i]
        price_below_ema = close[i] < ema_20[i]
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if price_above_ema and trend_aligned[i] == 1 and vol_spike:
                # Long: break above EMA20 in uptrend
                signals[i] = 0.25
                position = 1
            elif price_below_ema and trend_aligned[i] == -1 and vol_spike:
                # Short: break below EMA20 in downtrend
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses below EMA20 OR trend turns down
                if close[i] < ema_20[i] or trend_aligned[i] == -1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above EMA20 OR trend turns up
                if close[i] > ema_20[i] or trend_aligned[i] == 1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals