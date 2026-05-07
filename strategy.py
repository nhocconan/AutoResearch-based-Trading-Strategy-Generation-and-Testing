#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_1dTrend_Volume"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's data
    # Use previous day's high, low, close for today's levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    valid_idx = ~(np.isnan(prev_high) | np.isnan(prev_low) | np.isnan(prev_close))
    prev_high = np.where(valid_idx, prev_high, prev_close)
    prev_low = np.where(valid_idx, prev_low, prev_close)
    prev_close = np.where(valid_idx, prev_close, prev_close)
    
    # Camarilla calculations
    pp = (prev_high + prev_low + prev_close) / 3
    r3 = prev_close + (prev_high - prev_low) * 1.1
    s3 = prev_close - (prev_high - prev_low) * 1.1
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 20-period average (approx 10 hours)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Wait for volume MA and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks below S3 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] < s3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks above R3 with volume and daily downtrend
            elif close[i] > r3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns above S3 or volume drops
            if close[i] > s3_aligned[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns below R3 or volume drops
            if close[i] < r3_aligned[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with daily trend and volume confirmation
# - Camarilla R3/S3 act as strong support/resistance levels derived from prior day
# - Break below S3 with volume in daily uptrend = long opportunity (mean reversion in uptrend)
# - Break above R3 with volume in daily downtrend = short opportunity (mean reversion in downtrend)
# - Volume spike (2x average) confirms institutional participation
# - Works in both bull (buy S3 breaks in uptrend) and bear (sell R3 breaks in downtrend)
# - Exit when price returns to S3/R3 level or volume weakens
# - Position size 0.25 targets ~25-40 trades/year, avoiding fee drag
# - Daily EMA(34) filter ensures alignment with higher timeframe trend
# - Camarilla levels provide precise institutional-grade support/resistance
# - Fewer conditions (3: price level, volume, trend) reduce overtrading risk