#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily high, low, close for Camarilla calculation
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla levels (R3, S3)
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    camarilla_range = daily_high - daily_low
    r3 = daily_close + camarilla_range * 1.1 / 2.0
    s3 = daily_close - camarilla_range * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 4-period average (2 days of 12h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above S3 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below R3 with volume and daily downtrend
            elif close[i] < r3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price back below S3 or volume drops
            if close[i] < s3_aligned[i] or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price back above R3 or volume drops
            if close[i] > r3_aligned[i] or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with daily trend and volume confirmation
# - Camarilla R3/S3 act as strong support/resistance levels derived from prior day's range
# - Breakout above S3 with volume in daily uptrend = long opportunity
# - Breakdown below R3 with volume in daily downtrend = short opportunity
# - Volume spike (2x average) confirms institutional participation
# - Works in both bull (buy S3 breaks in uptrend) and bear (sell R3 breaks in downtrend)
# - Exit when price returns to S3/R3 or volume weakens
# - Position size 0.30 targets 15-35 trades/year, avoiding excessive fee drag
# - Daily EMA(34) filter ensures alignment with intermediate-term trend
# - Camarilla levels provide institutional-grade support/resistance that works across market regimes