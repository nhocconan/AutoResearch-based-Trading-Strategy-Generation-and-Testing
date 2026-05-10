#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend
Hypothesis: Price breaks Camarilla R1 (long) or S1 (short) levels from prior day's range, with 4h EMA50 trend filter and volume confirmation.
1h timeframe for precise entry timing; 4h trend filter to avoid counter-trend trades.
Volume confirmation ensures breakouts have institutional participation.
Designed for 15-30 trades/year on 1h to minimize fee drag while capturing directional moves in bull/bear markets.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels from prior day: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema50_4h[49] = np.mean(close_4h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema50_4h[i-1]
    
    # 4h volume SMA20 for volume confirmation
    vol_sma20_4h = np.full(len(df_4h), np.nan)
    if len(df_4h) >= 20:
        vol_sma20_4h[19] = np.mean(df_4h['volume'].values[:20])
        for i in range(20, len(df_4h)):
            vol_sma20_4h[i] = (vol_sma20_4h[i-1] * 19 + df_4h['volume'].values[i]) / 20
    
    # Align 1d and 4h indicators to 1h
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    vol_sma20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_sma20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_sma20_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average 4h volume (scaled)
        vol_4h_scaled = vol_sma20_4h_aligned[i] / 4.0  # 4x 1h bars in 4h
        volume_confirm = volume[i] > 1.5 * vol_4h_scaled
        
        # Trend and price relative to Camarilla levels
        is_uptrend = close[i] > ema50_4h_aligned[i]
        is_downtrend = close[i] < ema50_4h_aligned[i]
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1, in uptrend, with volume
            if price_above_r1 and is_uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1, in downtrend, with volume
            elif price_below_s1 and is_downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price falls back below R1 or trend turns down
            if not price_above_r1 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price rises back above S1 or trend turns up
            if not price_below_s1 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals