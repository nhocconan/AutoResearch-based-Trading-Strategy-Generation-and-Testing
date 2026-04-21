#!/usr/bin/env python3
"""
1h_RSI2_CR14_4hTrend_VolumeConfirm_v1
Hypothesis: RSI(2) for mean reversion on 1h with 4h EMA50 trend filter and volume spike confirmation.
Trades on pullbacks in trending markets, designed for low frequency (15-30/year) to avoid fee drag.
Uses 1h primary with 4h HTF for trend context - works in both bull (buy dips) and bear (sell rallies).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # === 4h trend filter: 50-period EMA ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 4h volume average (20-period) for spike detection ===
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h[np.isnan(vol_ma_4h)] = 1.0  # avoid division by zero
    vol_ratio_4h = volume_4h / vol_ma_4h
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # === RSI(2) on 1h for mean reversion signals ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_2 = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ratio_4h_aligned[i]) or
            np.isnan(rsi_2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_4h = ema_50_4h_aligned[i]
        vol_spike = vol_ratio_4h_aligned[i]
        rsi_val = rsi_2[i]
        
        if position == 0:
            # Long: RSI2 < 10 (oversold) + price above 4h EMA50 + volume spike > 1.5
            if rsi_val < 10 and price_close > trend_4h and vol_spike > 1.5:
                signals[i] = 0.20
                position = 1
            # Short: RSI2 > 90 (overbought) + price below 4h EMA50 + volume spike > 1.5
            elif rsi_val > 90 and price_close < trend_4h and vol_spike > 1.5:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit on RSI mean reversion: Long exits at RSI2 > 50, Short exits at RSI2 < 50
            if position == 1 and rsi_val > 50:
                signals[i] = 0.0
                position = 0
            elif position == -1 and rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI2_CR14_4hTrend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0