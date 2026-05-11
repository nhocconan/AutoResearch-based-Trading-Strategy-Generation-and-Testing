#!/usr/bin/env python3
"""
1h_4h1d_TrendReversal_v1
Hypothesis: Combines 4h trend direction (via EMA50) with 1d mean reversion signals
(RSI extremes) and volume confirmation on 1h timeframe. Uses 4h for primary trend
direction to avoid counter-trend trades, 1d RSI for overextension signals, and
1h for precise entry timing. Session filter (08-20 UTC) reduces noise. Designed
for low trade frequency (target: 15-35 trades/year) by requiring confluence of
trend, mean reversion, and volume. Works in bull/bear markets by following
4h trend while fading 1d extremes.
"""

name = "1h_4h1d_TrendReversal_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for RSI mean reversion
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4h EMA50 for trend direction ---
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # --- 1d RSI(14) for mean reversion ---
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # --- 1h volume spike detection (20-period) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma!=0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for indicators)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_14_aligned[i]) or
            np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: maintain current position or flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: 4h uptrend (price > EMA50) + 1d oversold (RSI < 30) + volume spike
            if (close[i] > ema_50_4h_aligned[i] and
                rsi_14_aligned[i] < 30 and
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend (price < EMA50) + 1d overbought (RSI > 70) + volume spike
            elif (close[i] < ema_50_4h_aligned[i] and
                  rsi_14_aligned[i] > 70 and
                  volume_spike):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: 4h trend breaks down OR 1d RSI returns to neutral (>50)
                if (close[i] < ema_50_4h_aligned[i] or
                    rsi_14_aligned[i] > 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: 4h trend breaks up OR 1d RSI returns to neutral (<50)
                if (close[i] > ema_50_4h_aligned[i] or
                    rsi_14_aligned[i] < 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals