#!/usr/bin/env python3
# 6H_PivotBreakoutWithMomentumAndVolume
# Hypothesis: 6-hour timeframe with 1-day pivot levels (R3/S3) for breakout/breakdown, confirmed by momentum (RSI > 50/ < 50) and volume spike (> 2x 20-period average). Works in bull markets (breakouts continue with momentum) and bear markets (mean reversion from extremes via short entries at R3/S3 with bearish momentum). Target: 50-150 total trades over 4 years = 12-37/year.

name = "6H_PivotBreakoutWithMomentumAndVolume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Momentum: RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 1d data for pivot levels (R3/S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for pivot calculation
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    # Pivot point (PP) and support/resistance levels
    pp = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    r3 = pp + 2 * (prev_high_1d - prev_low_1d)
    s3 = pp - 2 * (prev_high_1d - prev_low_1d)
    
    # Align 1d pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup for RSI
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + volume spike + RSI > 50 (bullish momentum)
            if (close[i] > r3_aligned[i] and 
                volume_spike[i] and 
                rsi[i] > 50):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + volume spike + RSI < 50 (bearish momentum)
            elif (close[i] < s3_aligned[i] and 
                  volume_spike[i] and 
                  rsi[i] < 50):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous day's range (between S3 and R3) OR RSI < 40 (loss of momentum)
            if (close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or \
               rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters previous day's range (between S3 and R3) OR RSI > 60 (loss of bearish momentum)
            if (close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or \
               rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals