#!/usr/bin/env python3
"""
4h_RSI2_MeanReversion_1dTrend_VolumeFilter_v1
Hypothesis: RSI(2) extreme mean reversion with 1d EMA50 trend filter and volume spike confirmation.
Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends.
Low trade frequency expected due to strict RSI(2) <10 or >90 thresholds.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === RSI(2) on 4h close ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_1d = ema_50_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: RSI(2) < 10 (extreme oversold) + volume spike > 1.5 + price above 1d EMA50
            if rsi_val < 10 and vol_spike > 1.5 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90 (extreme overbought) + volume spike > 1.5 + price below 1d EMA50
            elif rsi_val > 90 and vol_spike > 1.5 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when RSI returns to neutral zone (40-60) or opposite extreme
            if position == 1 and (rsi_val >= 40 or rsi_val > 90):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (rsi_val <= 60 or rsi_val < 10):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_RSI2_MeanReversion_1dTrend_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0