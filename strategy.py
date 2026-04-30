#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above R1, price > 4h EMA50, and volume > 2.0x 20-bar avg.
# Short when price breaks below S1, price < 4h EMA50, and volume > 2.0x 20-bar avg.
# Exit when price reverts to the Camarilla pivot point (mean reversion).
# Uses 4h EMA50 for higher timeframe trend alignment, targeting 15-37 trades/year on 1h.
# Trend filter avoids counter-trend trades, volume confirmation reduces false signals.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend alignment.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 40:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels from previous 4h bar (using same 4h data)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r1 = close_4h + (range_4h * 1.1 / 12)
    s1 = close_4h - (range_4h * 1.1 / 12)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(volume_confirm[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_pivot = pivot_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R1, price > 4h EMA50, volume spike, in session
            if (curr_close > curr_r1 and 
                curr_close > curr_ema_50_4h and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1, price < 4h EMA50, volume spike, in session
            elif (curr_close < curr_s1 and 
                  curr_close < curr_ema_50_4h and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price reverts to pivot point (mean reversion)
            if curr_close <= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit condition: price reverts to pivot point (mean reversion)
            if curr_close >= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals