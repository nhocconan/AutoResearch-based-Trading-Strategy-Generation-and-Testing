#!/usr/bin/env python3
# 4H_CAMARILLA_R3_S3_BREAKOUT_1D_VOLUME_SPIKE
# Hypothesis: Camarilla pivot levels (R3/S3) from daily timeframe identify institutional support/resistance.
# Breakouts above R3 or below S3 with volume confirmation (volume > 1.5x 20-period average) capture
# institutional breakout moves. Trend filter uses 50-period EMA on 1d to avoid counter-trend trades.
# Works in bull markets (breakouts above R3 in uptrend) and bear markets (breakdowns below S3 in downtrend).
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years).

name = "4H_CAMARILLA_R3_S3_BREAKOUT_1D_VOLUME_SPIKE"
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
    volume = prices['volume'].values
    
    # Daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's close for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = np.nan  # First day has no previous close
    
    # Calculate Camarilla levels for each day
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    rang = high_1d - low_1d
    r3 = close_1d + rang * 1.1 / 2
    s3 = close_1d - rang * 1.1 / 2
    
    # EMA50 for trend filter
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values  # Using price as proxy for volume calculation
    vol_ratio = volume / np.where(pd.Series(close).rolling(window=20, min_periods=1).mean().values > 0, 
                                  pd.Series(close).rolling(window=20, min_periods=20).mean().values, 1)
    volume_spike = volume > (1.5 * pd.Series(volume).rolling(window=20, min_periods=20).mean().values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one day of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume spike in uptrend
            if (close[i] > r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume spike in downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or trend reversal
            if (close[i] < r3_aligned[i] or 
                close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or trend reversal
            if (close[i] > s3_aligned[i] or 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals