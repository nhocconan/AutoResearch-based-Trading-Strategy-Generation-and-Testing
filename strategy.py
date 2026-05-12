#!/usr/bin/env python3
"""
6h_1d_1w_Keltner_Breakout_VolumeTrend
Hypothesis: 6-hour breakouts from Keltner Channel (ATR-based) with 1d trend filter and volume confirmation.
Keltner Channel adapts to volatility, providing dynamic support/resistance that works in both bull and bear markets.
Long when price breaks above upper Keltner band with volume spike and 1d uptrend.
Short when price breaks below lower Keltner band with volume spike and 1d downtrend.
Uses weekly context via 1d EMA200 filter to avoid counter-trend trades in strong trends.
Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""

name = "6h_1d_1w_Keltner_Breakout_VolumeTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.8x 40-period average (on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # ATR for Keltner Channel (20-period, 2.0 multiplier)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Keltner Channel: EMA20 center, ± ATR*2
    ema_center = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema_center + (2.0 * atr)
    keltner_lower = ema_center - (2.0 * atr)
    
    # 1d data for trend filter and weekly context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d EMA200 for weekly trend filter (avoid counter-trend)
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Keltner + volume spike + price above 1d EMA50 + above EMA200
            if (close[i] > keltner_upper[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i] and
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Keltner + volume spike + price below 1d EMA50 + below EMA200
            elif (close[i] < keltner_lower[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Keltner channel OR closes below 1d EMA50
            if (keltner_lower[i] < close[i] < keltner_upper[i]) or \
               close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Keltner channel OR closes above 1d EMA50
            if (keltner_lower[i] < close[i] < keltner_upper[i]) or \
               close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals