#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Combines weekly trend filter with Camarilla R3/S3 level breaks on 12h timeframe.
# Goes long when price breaks above R3 with weekly uptrend and volume confirmation.
# Goes short when price breaks below S3 with weekly downtrend and volume confirmation.
# Uses weekly trend to avoid counter-trend trades, reducing whipsaw in bear markets.
# Targets 12-37 trades/year to minimize fee drag while capturing strong momentum moves.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(34) for trend direction
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for current day using previous day's OHLC
    camarilla_r3 = np.zeros(len(prices))
    camarilla_s3 = np.zeros(len(prices))
    
    for i in range(len(prices)):
        # Get previous day's OHLC (need to find corresponding daily bar)
        # Since we're on 12h timeframe, we can use the most recent completed daily bar
        if i >= 2:  # Need at least 2 bars to have previous day
            # For 12h chart, each day has 2 bars
            prev_day_idx = (i // 2) * 2 - 2  # Index of first bar of previous day
            if prev_day_idx >= 0 and prev_day_idx + 1 < len(df_1d):
                # Use previous day's OHLC
                phigh = df_1d['high'].iloc[prev_day_idx // 2] if prev_day_idx // 2 < len(df_1d) else df_1d['high'].iloc[-1]
                plow = df_1d['low'].iloc[prev_day_idx // 2] if prev_day_idx // 2 < len(df_1d) else df_1d['low'].iloc[-1]
                pclose = df_1d['close'].iloc[prev_day_idx // 2] if prev_day_idx // 2 < len(df_1d) else df_1d['close'].iloc[-1]
            else:
                # Fallback to most recent available
                idx = min(len(df_1d) - 1, max(0, (i // 2) - 1))
                phigh = df_1d['high'].iloc[idx]
                plow = df_1d['low'].iloc[idx]
                pclose = df_1d['close'].iloc[idx]
            
            # Camarilla calculations
            range_val = phigh - plow
            camarilla_r3[i] = pclose + range_val * 1.1 / 4
            camarilla_s3[i] = pclose - range_val * 1.1 / 4
        else:
            # Not enough data yet
            camarilla_r3[i] = 0
            camarilla_s3[i] = 0
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with weekly uptrend and volume
            if (close[i] > camarilla_r3[i] and 
                close[i-1] <= camarilla_r3[i] and 
                close[i] > ema_34_1w_aligned[i] and  # Weekly uptrend filter
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with weekly downtrend and volume
            elif (close[i] < camarilla_s3[i] and 
                  close[i-1] >= camarilla_s3[i] and 
                  close[i] < ema_34_1w_aligned[i] and  # Weekly downtrend filter
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S3 (reversal signal) or weekly trend changes
            if close[i] < camarilla_s3[i] and close[i-1] >= camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_34_1w_aligned[i]:  # Weekly trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above R3 (reversal signal) or weekly trend changes
            if close[i] > camarilla_r3[i] and close[i-1] <= camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_34_1w_aligned[i]:  # Weekly trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals