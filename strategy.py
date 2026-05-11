#!/usr/bin/env python3
# 1h_4hTrend_1dVolatilityBreakout
# Hypothesis: Use 4h trend (via 20 EMA) for directional bias and 1d volatility expansion (ATR breakout) for entries on 1h.
# Long when: 4h EMA20 rising, 1h close breaks above 1h open + 0.5*1d ATR, volume > 1.2x 20-period average.
# Short when: 4h EMA20 falling, 1h close breaks below 1h open - 0.5*1d ATR, volume > 1.2x 20-period average.
# Exit when 4h EMA20 trend reverses or price crosses back to 1h open.
# Works in bull by catching trend continuations and in bear by catching breakdowns with trend filter.
# Volatility breakout captures momentum bursts, 4h EMA filters counter-trend noise.

name = "1h_4hTrend_1dVolatilityBreakout"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for EMA20 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for ATR(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4h EMA20 trend ---
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    for i in range(20, len(close_4h)):
        if i == 20:
            ema_4h[i] = np.mean(close_4h[0:20])
        else:
            ema_4h[i] = (close_4h[i] * 2 / (20 + 1)) + (ema_4h[i-1] * (19 / (20 + 1)))
    
    # EMA slope (rising/falling)
    ema_slope = np.full(len(close_4h), np.nan)
    for i in range(21, len(close_4h)):
        ema_slope[i] = ema_4h[i] - ema_4h[i-1]
    
    # Align 4h EMA and slope to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_slope)
    
    # --- 1d ATR(14) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr = np.zeros(len(close_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            atr_1d[i] = np.mean(tr[0:14])
        else:
            atr_1d[i] = (tr[i] * 1 + atr_1d[i-1] * 13) / 14  # Wilder smoothing
    
    # Align 1d ATR to 1h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 4h EMA20, 1d ATR14, and volume MA20
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_4h_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility breakout conditions
        breakout_up = close[i] > open_[i] + 0.5 * atr_1d_aligned[i]
        breakout_down = close[i] < open_[i] - 0.5 * atr_1d_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.2  # 20% above average
        
        if position == 0:
            if breakout_up and ema_slope_aligned[i] > 0 and vol_spike:
                # Long: upward volatility breakout + rising 4h EMA20 + volume spike
                signals[i] = 0.20
                position = 1
            elif breakout_down and ema_slope_aligned[i] < 0 and vol_spike:
                # Short: downward volatility breakout + falling 4h EMA20 + volume spike
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Exit long: 4h EMA20 trend turns negative OR price falls back to open
                if ema_slope_aligned[i] < 0 or close[i] < open_[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: 4h EMA20 trend turns positive OR price rises back to open
                if ema_slope_aligned[i] > 0 or close[i] > open_[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals