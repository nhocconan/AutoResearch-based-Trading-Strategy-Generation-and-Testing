# SPDX-FileCopyrightText: 2025 AlpacaKC
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
Hypothesis: Camarilla pivot level breakout with 12h trend filter and volume confirmation.
Long when price breaks above R1 (H5) in 12h uptrend with volume > 1.5x avg.
Short when price breaks below S1 (L5) in 12h downtrend with volume > 1.5x avg.
Exit on Camarilla pivot level touch (PP level) or 12h trend reversal.
Designed for mean-reversion in ranging markets and breakout in trending markets.
Target: 25-40 trades/year to minimize fee drag while capturing both regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate daily high/low/close for Camarilla pivots (use previous day)
    # Since we're on 4h timeframe, we need daily pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: H5 = Close + 1.1*(High-Low)*1.1/2, L5 = Close - 1.1*(High-Low)*1.1/2
    # Actually standard Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_R1 = np.full(len(close_1d), np.nan)
    camarilla_S1 = np.full(len(close_1d), np.nan)
    camarilla_PP = np.full(len(close_1d), np.nan)  # Pivot Point
    
    for i in range(1, len(close_1d)):
        hl_range = high_1d[i-1] - low_1d[i-1]
        camarilla_PP[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3
        camarilla_R1[i] = close_1d[i-1] + (hl_range * 1.1 / 12)
        camarilla_S1[i] = close_1d[i-1] - (hl_range * 1.1 / 12)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * multiplier) + (ema_12h[i-1] * (1 - multiplier))
    
    # Calculate 12h volume moving average (20-period)
    volume_12h = df_12h['volume'].values
    vol_ma_period = 20
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    for i in range(vol_ma_period, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-vol_ma_period:i+1])
    
    # Align all 1d and 12h indicators to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_PP_aligned = align_htf_to_ltf(prices, df_1d, camarilla_PP)
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # 4h volume confirmation (20-period average)
    vol_ma_4h_period = 20
    vol_ma_4h = np.full(n, np.nan)
    for i in range(vol_ma_4h_period, n):
        vol_ma_4h[i] = np.mean(volume[i-vol_ma_4h_period:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(1, 50, 20, 20)  # Camarilla needs 1 day, EMA(50), vol MA(20) both timeframes
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_R1_aligned[i]) or
            np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(camarilla_PP_aligned[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_4h[i] if vol_ma_4h[i] > 0 else 0
        
        # Trend filter: price above/below 12h EMA50
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average 4h volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above Camarilla R1 in uptrend with volume
            if uptrend and volume_confirmation and price > camarilla_R1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 in downtrend with volume
            elif downtrend and volume_confirmation and price < camarilla_S1_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price touches Camarilla PP or trend reverses
            if price <= camarilla_PP_aligned[i] or price < ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price touches Camarilla PP or trend reverses
            if price >= camarilla_PP_aligned[i] or price > ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0