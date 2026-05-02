#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Camarilla levels derived from prior 1d range provide institutional support/resistance
# Breakout of R3 (short) or S3 (long) with volume > 2x average and 1d trend alignment
# Works in bull/bear by capturing institutional breakouts with volume confirmation
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_Camarilla_R3_S3_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot calculation (prior day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on prior 1d range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    rang = high_1d - low_1d
    camarilla_s3 = close_1d - 1.1 * rang  # S3 level
    camarilla_r3 = close_1d + 1.1 * rang   # R3 level
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above S3 with bullish 1d trend and volume spike
            if (close[i] > camarilla_s3_aligned[i] and 
                close[i-1] <= camarilla_s3_aligned[i-1] and  # Just broke above
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below R3 with bearish 1d trend and volume spike
            elif (close[i] < camarilla_r3_aligned[i] and 
                  close[i-1] >= camarilla_r3_aligned[i-1] and  # Just broke below
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price closes below S3 OR trend turns bearish
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price closes above R3 OR trend turns bullish
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals