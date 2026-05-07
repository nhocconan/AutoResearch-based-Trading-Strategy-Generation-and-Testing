#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: On 4h chart, breakout above Camarilla R3 or below S3 with volume spike and 12h EMA50 trend filter.
# Uses Camarilla levels from prior day for structure, volume confirmation to filter noise, and 12h EMA for trend alignment.
# Designed for moderate trade frequency (~20-40/year) to balance opportunity and fee drag in bull/bear markets.
timeframe = "4h"
name = "4h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_ = high - low
    if range_ == 0:
        return close, close, close, close
    c = close
    h = high
    l = low
    r3 = c + (h - l) * 1.1 / 2
    s3 = c - (h - l) * 1.1 / 2
    return r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    ema_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for warmup
        # Skip if any critical value is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day's OHLC
        if i < 24:  # Need at least 24 periods (6 hours * 4 = 1 day) for prior day
            continue
            
        # Get previous day's high, low, close (24 periods back)
        prev_day_high = np.max(high[i-24:i])
        prev_day_low = np.min(low[i-24:i])
        prev_day_close = close[i-1]  # Previous close
        
        r3, s3 = calculate_camarilla(prev_day_high, prev_day_low, prev_day_close)
        
        if position == 0:
            # Long: price breaks above R3 + 12h uptrend + volume spike
            if close[i] > r3 and ema_12h_aligned[i] > ema_12h_aligned[i-1] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + 12h downtrend + volume spike
            elif close[i] < s3 and ema_12h_aligned[i] < ema_12h_aligned[i-1] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S3 or 12h trend turns down
            if close[i] < s3:
                signals[i] = 0.0
                position = 0
            elif ema_12h_aligned[i] < ema_12h_aligned[i-1]:  # 12h trend turns down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R3 or 12h trend turns up
            if close[i] > r3:
                signals[i] = 0.0
                position = 0
            elif ema_12h_aligned[i] > ema_12h_aligned[i-1]:  # 12h trend turns up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals