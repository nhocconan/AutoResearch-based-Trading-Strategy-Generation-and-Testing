# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R3/S3 levels provide strong support/resistance in 1d timeframe; breakout with volume confirmation and 1d trend alignment gives high-probability trades. Works in bull markets by catching breakouts and in bear markets by catching reversals at key levels.
# Timeframe: 12h (as required). Uses 1d Camarilla levels and 1w trend filter.
# Expected trades: ~20-40/year to stay within fee limits.

#!/usr/bin/env python3

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (based on previous day)
    # Camarilla R3 = close + (high - low) * 1.1/2
    # Camarilla S3 = close - (high - low) * 1.1/2
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        rang = prev_high - prev_low
        camarilla_r3[i] = prev_close + rang * 1.1 / 2
        camarilla_s3[i] = prev_close - rang * 1.1 / 2
    
    # Get 1w data for trend filter (use EMA34 on weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * 2/35) + (ema_34_1w[i-1] * 33/35)
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Align 1w EMA34 to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume average (24-period ~ 12 days) for volume spike filter
    vol_ma_24 = np.full_like(volume, np.nan)
    for i in range(23, len(volume)):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Warmup period
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 24-period average
        vol_spike = volume[i] > 2.0 * vol_ma_24[i]
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 + volume spike + 1w uptrend (close > EMA34)
            if (close[i] > camarilla_r3_aligned[i] and vol_spike and 
                close_1w[-1] > ema_34_1w_aligned[i] if len(close_1w) > 0 else False):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 + volume spike + 1w downtrend (close < EMA34)
            elif (close[i] < camarilla_s3_aligned[i] and vol_spike and 
                  close_1w[-1] < ema_34_1w_aligned[i] if len(close_1w) > 0 else False):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Camarilla R3 or loss of volume/spike
            if (close[i] < camarilla_r3_aligned[i] or not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Camarilla S3 or loss of volume/spike
            if (close[i] > camarilla_s3_aligned[i] or not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals