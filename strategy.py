#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 5 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h Camarilla levels: R3, S3 from previous day
    # Camarilla: R3 = close + (high - low) * 1.1/2, S3 = close - (high - low) * 1.1/2
    prev_close = df_12h['close'].shift(1).values
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (already on 12h, no alignment needed for same TF)
    # But we need to shift by 1 because levels are based on previous day
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 12h volume spike: > 2.0x 10-period average (~5 days)
    vol_ma_12h = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_spike_12h = volume > 2.0 * vol_ma_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(10, 34)  # Wait for volume MA and EMA34
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with volume spike and 1d uptrend
            if close[i] > camarilla_r3[i] and vol_spike_12h[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: Break below S3 with volume spike and 1d downtrend
            elif close[i] < camarilla_s3[i] and vol_spike_12h[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: Price below S3 or trend reversal
            if close[i] < camarilla_s3[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: Price above R3 or trend reversal
            if close[i] > camarilla_r3[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Note: Camarilla levels calculated from previous 12h period's high/low/close.
# Volume spike threshold set to 2.0x to ensure only strong breakouts trigger entries.
# Position size 0.30 limits risk per trade. Exit on retrace to S3/R3 or trend reversal.
# Designed for 12h timeframe with target of 50-150 total trades over 4 years (12-37/year).