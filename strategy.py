#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike
# Hypothesis: Uses Camarilla R3/S3 breakouts with weekly trend filter and volume spikes for high-probability entries.
# Weekly trend (1w EMA34) filters direction to avoid counter-trend trades. Volume > 2.0x 20-period MA confirms momentum.
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year). Works in bull/bear by aligning with weekly trend.
# Position size 0.25 for balanced risk management.

name = "6h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot levels (using previous day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR for volatility filter (optional but good practice)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day's OHLC (R3 and S3 - wider bands)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # R3 and S3 levels (wider than R1/S1 for fewer, higher-quality signals)
    r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get weekly EMA for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume average for confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 14)  # Warmup for volume MA, weekly EMA, and ATR
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation and volatility filter
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        volatility_filter = atr[i] > 0  # Ensure valid ATR
        
        if position == 0:
            # Long entry: price breaks above R3 with volume confirmation, weekly uptrend
            if close[i] > r3_aligned[i] and volume_confirm and uptrend and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume confirmation, weekly downtrend
            elif close[i] < s3_aligned[i] and volume_confirm and downtrend and volatility_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls back below R3 or weekly trend turns down
            if close[i] < r3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above S3 or weekly trend turns up
            if close[i] > s3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals