#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
# Uses Camarilla pivot levels from daily data to identify institutional support/resistance.
# In trending markets, price breaks beyond R3/S3 with continuation.
# Uses 1d EMA34 for trend direction and volume spike for confirmation.
# Designed to work in both bull (breakouts above R3) and bear (breakouts below S3) markets.
# Target: 20-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Calculate EMA(34) on daily close for trend filter
    ema_34_1d = np.full(len(close_1d), np.nan)
    alpha = 2 / (34 + 1)
    for i in range(len(close_1d)):
        if i < 33:
            ema_34_1d[i] = np.mean(close_1d[:i+1]) if i > 0 else close_1d[i]
        else:
            if np.isnan(ema_34_1d[i-1]):
                ema_34_1d[i] = np.mean(close_1d[i-33:i+1])
            else:
                ema_34_1d[i] = close_1d[i] * alpha + ema_34_1d[i-1] * (1 - alpha)
    
    # Align Camarilla levels and EMA to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from daily EMA34
        # Use previous bar's EMA to avoid look-ahead
        if i > 0 and not np.isnan(ema_34_1d_aligned[i-1]):
            trend_up = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            trend_down = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: price breaks above R3 + uptrend + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + downtrend + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price re-enters below R3 or trend turns down
            if (close[i] < camarilla_r3_aligned[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above S3 or trend turns up
            if (close[i] > camarilla_s3_aligned[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0