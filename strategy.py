#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Reversal with 1d EMA50 Trend Filter and Volume Confirmation
# Uses weekly Camarilla pivot levels (R3/S3, R4/S4) calculated from prior week OHLC
# In uptrend (price > 1d EMA50): buy at S3/S4 with volume spike, target R3/R4
# In downtrend (price < 1d EMA50): sell at R3/R4 with volume spike, target S3/S4
# Weekly pivots provide strong institutional levels; volume confirms participation
# Trend filter avoids counter-trend whipsaws; reversals at extreme pivots have high RR
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag

name = "6h_WeeklyPivot_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (based on prior week OHLC)
    # R4 = close + 1.5*(high - low), R3 = close + 1.0*(high - low)
    # S3 = close - 1.0*(high - low), S4 = close - 1.5*(high - low)
    wk_high = df_w['high'].values
    wk_low = df_w['low'].values
    wk_close = df_w['close'].values
    
    pivot_range = wk_high - wk_low
    r4 = wk_close + 1.5 * pivot_range
    r3 = wk_close + 1.0 * pivot_range
    s3 = wk_close - 1.0 * pivot_range
    s4 = wk_close - 1.5 * pivot_range
    
    # Align weekly pivot levels to 6h (wait for weekly close)
    r4_aligned = align_htf_to_ltf(prices, df_w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_w, s4)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price at S3/S4 (weekly support) in uptrend + volume spike
            if ((close[i] <= s3_aligned[i] * 1.002 or close[i] <= s4_aligned[i] * 1.002) and
                close[i] > ema_50_1d_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price at R3/R4 (weekly resistance) in downtrend + volume spike
            elif ((close[i] >= r3_aligned[i] * 0.998 or close[i] >= r4_aligned[i] * 0.998) and
                  close[i] < ema_50_1d_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches R3/R4 (weekly resistance) OR closes below 1d EMA50
            if close[i] >= r3_aligned[i] * 0.998 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches S3/S4 (weekly support) OR closes above 1d EMA50
            if close[i] <= s3_aligned[i] * 1.002 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals