# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Spike
# Hypothesis: Camarilla R3/S3 levels on 4h act as strong support/resistance. Breakout with volume spike and aligned with 1d trend (via EMA34) provides high-probability entries. Works in bull markets (breakouts above R3 in uptrend) and bear markets (breakdowns below S3 in downtrend). Volume spike filters weak breakouts. Target 20-50 trades/year to minimize fee drag.
# Timeframe: 4h, HTF: 1d

#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Spike"
timeframe = "4h"
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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 5 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h Camarilla levels: R3, S3 from previous day
    # Camarilla: R3 = close + (high - low) * 1.1/2, S3 = close - (high - low) * 1.1/2
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (already on 4h, no alignment needed for same TF)
    # But we need to shift by 1 because levels are based on previous day
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4h volume spike: > 2.0x 20-period average (~1.33 days)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = volume > 2.0 * vol_ma_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Wait for volume MA and EMA34
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with volume spike and 1d uptrend
            if close[i] > camarilla_r3[i] and vol_spike_4h[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume spike and 1d downtrend
            elif close[i] < camarilla_s3[i] and vol_spike_4h[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below S3 or trend reversal
            if close[i] < camarilla_s3[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above R3 or trend reversal
            if close[i] > camarilla_r3[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: Camarilla levels calculated from previous 4h period's high/low/close.
# Volume spike threshold set to 2.0x to ensure only strong breakouts trigger entries.
# Position size 0.25 limits risk per trade. Exit on retrace to S3/R3 or trend reversal.
# Designed for 20-50 trades/year on 4h timeframe.