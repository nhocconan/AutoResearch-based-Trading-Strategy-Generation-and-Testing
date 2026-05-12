#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dEMA50_Trend
# Hypothesis: On 4h timeframe, enter long when price closes above daily R3 with close > daily EMA50 and volume > 2x average.
# Enter short when price closes below daily S3 with close < daily EMA50 and volume > 2x average.
# Exit when price crosses daily EMA50 (trend reversal).
# Uses stronger S3/R3 levels for fewer, higher-quality trades. Works in bull markets via breakouts and in bear via short reversals at S3.

name = "4h_Camarilla_R3S3_Breakout_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivot calculation and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate pivot point and range
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    
    # Camarilla R3 and S3 levels (stronger than R1/S1)
    r3 = daily_pivot + daily_range * 1.1000
    s3 = daily_pivot - daily_range * 1.1000
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema1d_trend = ema50_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price closes above R3 with volume > 2x average and close > daily EMA50
            if close[i] > r3_val and close[i] > ema1d_trend and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S3 with volume > 2x average and close < daily EMA50
            elif close[i] < s3_val and close[i] < ema1d_trend and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below daily EMA50 (trend reversal)
            if close[i] < ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above daily EMA50 (trend reversal)
            if close[i] > ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals