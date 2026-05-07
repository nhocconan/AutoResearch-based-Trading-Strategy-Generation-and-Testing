#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_12hTrend_Volume"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Daily high/low/close for Camarilla calculation (from 12h data)
    # We'll use the 12h data to approximate daily levels (2 bars = 1 day)
    daily_high = pd.Series(high).rolling(window=2*6, min_periods=2*6).max().values  # 2 days of 12h bars
    daily_low = pd.Series(low).rolling(window=2*6, min_periods=2*6).min().values
    daily_close = pd.Series(close).rolling(window=2*6, min_periods=2*6).mean().values
    
    # Camarilla levels (R3, S3)
    r3 = daily_close + 1.1 * (daily_high - daily_low)
    s3 = daily_close - 1.1 * (daily_high - daily_low)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # 12h trend filter: EMA(34) on 12h close
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike detection: 6-period average (3 days of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 6, 2*6)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S3 with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_6[i] * 2.0
            uptrend = ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]
            
            if close[i] > s3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R3 with volume and 12h downtrend
            elif close[i] < r3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S3 or volume drops
            if close[i] < s3_aligned[i] or volume[i] < vol_ma_6[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R3 or volume drops
            if close[i] > r3_aligned[i] or volume[i] < vol_ma_6[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h trend and volume confirmation
# - Camarilla R3/S3 levels act as strong support/resistance (derived from daily range)
# - Breakout above S3 with volume in 12h uptrend = long opportunity
# - Breakdown below R3 with volume in 12h downtrend = short opportunity
# - Volume spike (2x average) confirms institutional participation
# - Works in both bull (buy S3 breaks in uptrend) and bear (sell R3 breaks in downtrend)
# - Exit when price returns to opposite Camarilla level (S3 for long, R3 for short) or volume weakens
# - Position size 0.25 targets 20-40 trades/year, avoiding fee drag
# - Uses 12h EMA(34) for trend filter to avoid whipsaws in ranging markets