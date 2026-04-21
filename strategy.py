#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h/1d moving average crossovers for trend direction and volume spike for entry timing.
In both bull and bear markets, price tends to move in the direction of higher timeframe trend after pullbacks.
Combines 4h EMA(21) and 1d EMA(50) for trend alignment, with 1h volume spike and price retracement to VWAP for entry.
Uses higher timeframes for signal direction (low frequency) and 1h for entry timing to reduce overtrading.
Target: 15-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(21) for intermediate trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA(50) for long-term trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Trend alignment: both 4h and 1d EMA agree on direction
    trend_up = ema_4h_aligned > ema_1d_aligned  # Bullish when 4h above 1d
    trend_down = ema_4h_aligned < ema_1d_aligned  # Bearish when 4h below 1d
    
    # Volume confirmation: 1h volume / 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # VWAP(20) for entry timing
    vwap_20 = (pd.Series(prices['close'].values * prices['volume'].values).rolling(20, min_periods=20).sum() / 
               pd.Series(prices['volume'].values).rolling(20, min_periods=20).sum()).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(vwap_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 2.0  # Volume spike filter
        
        if position == 0:
            # Enter long: bullish trend alignment + volume spike + price at VWAP support
            if (trend_up[i] and 
                vol_ratio_val > vol_threshold and 
                price_close <= vwap_20[i] * 1.005):  # Allow small buffer above VWAP
                signals[i] = 0.20
                position = 1
            # Enter short: bearish trend alignment + volume spike + price at VWAP resistance
            elif (trend_down[i] and 
                  vol_ratio_val > vol_threshold and 
                  price_close >= vwap_20[i] * 0.995):  # Allow small buffer below VWAP
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: trend breaks down or volume dries up
            if position == 1 and not trend_up[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_EMATrend_VolumeSpike_VWAP"
timeframe = "1h"
leverage = 1.0