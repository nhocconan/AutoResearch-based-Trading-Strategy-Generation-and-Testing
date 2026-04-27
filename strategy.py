#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot breakout with 1w trend filter and volume confirmation.
# Camarilla pivots provide structured support/resistance levels (R3/S3 for reversal, R4/S4 for breakout).
# Weekly trend filter ensures we trade in direction of higher timeframe momentum.
# Volume confirmation filters false breakouts.
# Works in bull markets (buy R4 breakouts in uptrend) and bear markets (sell S4 breakdowns in downtrend).
# Target: 20-40 trades/year to stay within fee limits.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on weekly close
    ema_50_1w = np.full(len(df_1w), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_1w)):
        if i < 49:
            ema_50_1w[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_50_1w[i-1]):
                ema_50_1w[i] = np.mean(close_1w[i-49:i+1])
            else:
                ema_50_1w[i] = close_1w[i] * alpha + ema_50_1w[i-1] * (1 - alpha)
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = close + ((high - low) * 1.1 / 2)
    # R3 = close + ((high - low) * 1.1 / 4)
    # S3 = close - ((high - low) * 1.1 / 4)
    # S4 = close - ((high - low) * 1.1 / 2)
    camarilla_r4 = np.zeros(len(df_1d))
    camarilla_r3 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    camarilla_s4 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        rng = high_1d[i] - low_1d[i]
        camarilla_r4[i] = close_1d[i] + (rng * 1.1 / 2)
        camarilla_r3[i] = close_1d[i] + (rng * 1.1 / 4)
        camarilla_s3[i] = close_1d[i] - (rng * 1.1 / 4)
        camarilla_s4[i] = close_1d[i] - (rng * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (need 1 bar delay for daily close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume spike: current volume > 2.0 * 20-period average (higher threshold for fewer trades)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA50
        # Use previous bar's EMA to avoid look-ahead
        if i > 0 and not np.isnan(ema_50_1w_aligned[i-1]):
            trend_up = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            trend_down = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: price breaks above R4 + uptrend + volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S4 + downtrend + volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price drops below R3 (mean reversion) or trend turns down
            if (close[i] < camarilla_r3_aligned[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above S3 (mean reversion) or trend turns up
            if (close[i] > camarilla_s3_aligned[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1wEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0