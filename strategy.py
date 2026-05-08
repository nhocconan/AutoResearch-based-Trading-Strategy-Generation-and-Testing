#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Camarilla pivot levels (R3/S3) for breakouts with 1-day EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 in uptrend with volume surge; short when price breaks below S3 in downtrend with volume surge.
# Uses daily EMA(34) for trend direction and daily ATR(14) for dynamic pivot adjustments.
# Designed for low trade frequency (12-37/year) to minimize fee flood and capture sustained moves in both bull and bear markets.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1
    # S3 = Pivot - (H - L) * 1.1
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r3 = pivot + (high_1d - low_1d) * 1.1
    s3 = pivot - (high_1d - low_1d) * 1.1
    
    # 1d EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_1d[1:] > ema_34_1d[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1d index
    
    # 1d ATR(14) for volatility (optional dynamic adjustment)
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_close[0] = high_1d[0] - close_1d[0]  # First value
    low_close[0] = low_1d[0] - close_1d[0]    # First value
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align daily indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    # Volume confirmation: 12h volume > 2.0x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA(34)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above R3 in uptrend with volume
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                close[i] > r3_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: break below S3 in downtrend with volume
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  close[i] < s3_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below S3 or trend turns down
            if close[i] < s3_aligned[i] or trend_up_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above R3 or trend turns up
            if close[i] > r3_aligned[i] or trend_up_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals