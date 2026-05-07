#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
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
    
    # Load 12h and 1w data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_12h) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 12h volume spike: > 2.0x 20-period average
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume > 2.0 * vol_ma_12h
    
    # 12h EMA20 for entry filter
    ema20_12h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d Camarilla levels: R3, S3 from previous day
    # Camarilla: R3 = close + (high - low) * 1.1/2, S3 = close - (high - low) * 1.1/2
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    camarilla_r3_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 2
    camarilla_s3_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 2
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 34)  # Wait for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(camarilla_r3_1d_aligned[i]) or 
            np.isnan(camarilla_s3_1d_aligned[i]) or np.isnan(ema20_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with volume spike, bullish trend (price > EMA34), and price above EMA20
            if (close[i] > camarilla_r3_1d_aligned[i] and vol_spike_12h[i] and 
                close[i] > ema34_1w_aligned[i] and close[i] > ema20_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume spike, bearish trend (price < EMA34), and price below EMA20
            elif (close[i] < camarilla_s3_1d_aligned[i] and vol_spike_12h[i] and 
                  close[i] < ema34_1w_aligned[i] and close[i] < ema20_12h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below S3 or trend turns bearish (price < EMA34)
            if close[i] < camarilla_s3_1d_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above R3 or trend turns bullish (price > EMA34)
            if close[i] > camarilla_r3_1d_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h timeframe reduces trade frequency to avoid fee drag while capturing significant moves.
# Uses 1d Camarilla levels (R3/S3) for institutional support/resistance, 1w EMA34 for trend filter,
# and 12h volume spike for confirmation. Trades only in direction of higher timeframe trend.
# Position size 0.25 limits risk. Target: 12-37 trades/year (50-150 total over 4 years).