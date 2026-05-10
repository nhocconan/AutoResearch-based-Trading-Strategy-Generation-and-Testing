#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_1dTrend_Volume
Hypothesis: Fade at Camarilla R3/S3 levels (mean reversion) with 1d EMA34 trend filter and volume spike confirmation.
Works in bull markets by fading overextended rallies and in bear markets by fading panic drops.
Targets 15-25 trades/year per symbol with discrete position sizing to minimize fee drag.
"""

name = "6h_Camarilla_R3_S3_Fade_1dTrend_Volume"
timeframe = "6h"
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
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume SMA(20) for volume filter
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    # Calculate 1d EMA34 for trend filter (using HTF data)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 34)  # Ensure volume SMA and EMA are ready
    
    for i in range(start_idx, n):
        if np.isnan(atr[i]) or np.isnan(vol_sma[i]) or np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day
        if i >= 4:  # Need at least 4 bars (previous day's data on 6h chart)
            # Previous day's OHLC (assuming 4 bars per day on 6h chart)
            prev_day_start = max(0, i - 4)
            prev_day_high = np.max(high[prev_day_start:i])
            prev_day_low = np.min(low[prev_day_start:i])
            prev_day_close = close[i-1]
            
            # Camarilla levels
            range_val = prev_day_high - prev_day_low
            r3 = prev_day_close + (range_val * 1.1 / 4)
            s3 = prev_day_close - (range_val * 1.1 / 4)
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 0:
            # Long: Fade below S3 with uptrend and volume spike
            if close[i] < s3 and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Fade above R3 with downtrend and volume spike
            elif close[i] > r3 and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses above EMA34 or reaches opposite extreme
            if close[i] > ema_34_1d_aligned[i] or close[i] > (s3 + (r3 - s3) * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses below EMA34 or reaches opposite extreme
            if close[i] < ema_34_1d_aligned[i] or close[i] < (r3 - (r3 - s3) * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals