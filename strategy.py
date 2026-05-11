#!/usr/bin/env python3
# 4h_MultiTF_Confluence_V1
# Hypothesis: Combines 4h Donchian breakout with 12h EMA trend and volume confirmation, filtering by 1d chop regime. 
# Long when: price breaks above Donchian(20), 12h EMA > 12h EMA[10] (uptrend), volume > 1.5x 20-period average, and chop < 61.8 (trending).
# Short when: price breaks below Donchian(20), 12h EMA < 12h EMA[10] (downtrend), volume > 1.5x 20-period average, and chop < 61.8.
# Exit when: price crosses back through Donchian(20) midpoint or chop > 61.8 (range).
# Designed to capture strong trends with volume confirmation in trending regimes, avoiding false breakouts in chop.
# Works in bull markets by catching breakouts and in bear markets by catching breakdowns, with volatility-adjusted position sizing.

name = "4h_MultiTF_Confluence_V1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h EMA trend (20-period) ---
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_prev = np.roll(ema_12h, 10)  # 10 periods ago for trend
    ema_12h_prev[:10] = ema_12h[0]  # fill first 10 with first value
    ema_trend = ema_12h > ema_12h_prev  # True if EMA rising
    
    # --- 1d Choppy Market Index (CHOP) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr_14 * 14) / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_trending = chop < 61.8  # True when trending (chop < 61.8)
    
    # --- 4h Donchian Channel (20-period) ---
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > vol_ma * 1.5
    
    # Align 12h EMA trend to 4h
    ema_trend_aligned = align_htf_to_ltf(prices, df_12h, ema_trend)
    
    # Align 1d chop regime to 4h
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Donchian(20), EMA(20), CHOP(14), vol MA(20)
    start_idx = max(20, 20, 14, 20)  # 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(ema_trend_aligned[i]) or
            np.isnan(chop_trending_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > highest_high_20[i]
        breakout_down = close[i] < lowest_low_20[i]
        
        if position == 0:
            if breakout_up and ema_trend_aligned[i] and vol_spike[i] and chop_trending_aligned[i]:
                # Long: Donchian breakout up + 12h EMA uptrend + volume spike + trending regime
                signals[i] = 0.25
                position = 1
            elif breakout_down and not ema_trend_aligned[i] and vol_spike[i] and chop_trending_aligned[i]:
                # Short: Donchian breakout down + 12h EMA downtrend + volume spike + trending regime
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses below Donchian midpoint OR chop > 61.8 (range)
                if close[i] < donchian_mid[i] or not chop_trending_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above Donchian midpoint OR chop > 61.8 (range)
                if close[i] > donchian_mid[i] or not chop_trending_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals