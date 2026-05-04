#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1w EMA50 for trend direction and Camarilla pivot levels from 1w for entry/exit
# Volume confirmation requires 1.8x average volume to ensure strong participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 12h timeframe
# Works in both bull and bear markets by following the 1w trend direction and using Camarilla for structure
# Prioritizes BTC/ETH performance with SOL as secondary

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF indicators
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels from 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    # Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    # R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    # We need R3 and S3 for breakout
    camarilla_r3 = np.full(len(df_1w), np.nan)
    camarilla_s3 = np.full(len(df_1w), np.nan)
    camarilla_pp = np.full(len(df_1w), np.nan)  # pivot point
    
    for i in range(len(df_1w)):
        if i < 1:  # Need previous week's data
            continue
        # Use previous week's OHLC to calculate this week's Camarilla levels
        prev_high = high_1w[i-1]
        prev_low = low_1w[i-1]
        prev_close = close_1w_arr[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        camarilla_pp[i] = pivot
        camarilla_r3[i] = pivot + 1.1 * (prev_high - prev_low) * 1.1 / 4.0
        camarilla_s3[i] = pivot - 1.1 * (prev_high - prev_low) * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pp)
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Camarilla breakout with 1w trend filter
        # Long: Price breaks above R3 + volume spike + price above 1w EMA50 (uptrend)
        # Short: Price breaks below S3 + volume spike + price below 1w EMA50 (downtrend)
        if position == 0:
            if (close[i] > camarilla_r3_aligned[i] and volume_spike and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < camarilla_s3_aligned[i] and volume_spike and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 OR price below 1w EMA50 (trend change)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R3 OR price above 1w EMA50 (trend change)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals