#!/usr/bin/env python3
# 12h Camarilla R3/S3 Breakout with 1d Trend Filter and Volume Confirmation
# Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance. 
# Breakout above R3 or below S3 with volume confirmation and 1d trend alignment
# captures momentum moves while avoiding false breakouts in ranging markets.
# Works in both bull and bear markets by following the 1d trend direction.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

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
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Requires daily high, low, close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    ph = df_1d['high'].shift(1).values  # Previous day high
    pl = df_1d['low'].shift(1).values   # Previous day low
    pc = df_1d['close'].shift(1).values # Previous day close
    
    # Camarilla levels: R3/S3 = C ± (H-L)*1.1/2
    camarilla_r3 = pc + (ph - pl) * 1.1 / 2
    camarilla_s3 = pc - (ph - pl) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (previous day's levels available at 00:00 UTC)
    r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d trend filter: EMA34
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5 * average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if Camarilla levels or trend data not ready
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R3 with volume confirmation and 1d uptrend
            if (close[i] > r3_12h[i] and 
                volume[i] > vol_threshold[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume confirmation and 1d downtrend
            elif (close[i] < s3_12h[i] and 
                  volume[i] > vol_threshold[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below S3 (reversal) or opposite breakout
            if close[i] < s3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above R3 (reversal) or opposite breakout
            if close[i] > r3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals