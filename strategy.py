#!/usr/bin/env python3
"""
Hypothesis: 6h Weekly Camarilla Pivot Breakout with Volume Spike and Daily Trend Filter.
- Uses weekly Camarilla levels (R3/S3, R4/S4) from 1d HTF data to identify institutional support/resistance.
- Breakout beyond R4/S4 with volume > 2x 20-period average signals strong momentum continuation.
- 1d EMA34 trend filter ensures alignment with daily trend to avoid counter-trend whipsaws.
- Discrete position size 0.25 limits drawdown. Target: 15-30 trades/year (60-120 total over 4 years).
- Works in bull/bear via trend filter and volatility-adjusted breakout thresholds.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    # Weekly OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla pivot levels (based on weekly range)
    weekly_range = weekly_high - weekly_low
    camarilla_H3 = weekly_close + weekly_range * 1.1 / 4
    camarilla_L3 = weekly_close - weekly_range * 1.1 / 4
    camarilla_H4 = weekly_close + weekly_range * 1.1 / 2
    camarilla_L4 = weekly_close - weekly_range * 1.1 / 2
    
    # Align weekly Camarilla levels to 6h timeframe (wait for weekly close)
    H3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_H3, additional_delay_bars=0)
    L3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_L3, additional_delay_bars=0)
    H4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_H4, additional_delay_bars=0)
    L4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_L4, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 50)  # volume MA, 1d EMA, weekly pivot lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close > H4 AND price above 1d EMA34 AND volume spike
            if close[i] > H4_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close < L4 AND price below 1d EMA34 AND volume spike
            elif close[i] < L4_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < H3 OR price crosses below 1d EMA34
            if close[i] < H3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > L3 OR price crosses above 1d EMA34
            if close[i] > L3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyCamarilla_H4L4_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0