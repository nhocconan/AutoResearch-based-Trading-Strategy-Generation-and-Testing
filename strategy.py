#!/usr/bin/env python3
# 6h_VolumeSpike_Camarilla_Reversal
# Hypothesis: Fade extreme price moves at daily Camarilla S3/R3 levels on 6b timeframe.
# Enter long when price touches S3 with volume spike and RSI < 30 (oversold).
# Enter short when price touches R3 with volume spike and RSI > 70 (overbought).
# Exit when price reverts to the daily pivot point.
# Uses volume confirmation to avoid false breaks and RSI for momentum exhaustion.
# Works in ranging markets (mean reversion) and avoids strong trends via RSI extremes.
# Target: 20-40 trades/year per symbol for low friction and high win rate.

name = "6h_VolumeSpike_Camarilla_Reversal"
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
    
    # Load daily data for Camarilla pivot calculation and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Typical price (Pivot point)
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    
    # Camarilla levels: S3 and R3
    r3 = daily_pivot + daily_range * 1.25
    s3 = daily_pivot - daily_range * 1.25
    
    # Align daily levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
    # Calculate daily RSI(14) for momentum exhaustion signal
    delta = pd.Series(daily_close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (to avoid noise)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        pivot_val = pivot_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Price at S3, oversold RSI, volume spike
            if low[i] <= s3_val and rsi_val < 30 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R3, overbought RSI, volume spike
            elif high[i] >= r3_val and rsi_val > 70 and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot (mean reversion complete)
            if close[i] >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot (mean reversion complete)
            if close[i] <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals