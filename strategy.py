#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike
# Hypothesis: On 4h timeframe, enter long when price closes above daily R3 with close > 1d EMA34 and volume > 1.5x 20-period average.
# Enter short when price closes below daily S3 with close < 1d EMA34 and volume > 1.5x 20-period average.
# Exit when price crosses 1d EMA34 (trend reversal).
# Uses daily timeframe for stronger trend filter and volume spike for confirmation.
# Targets 20-40 trades/year for low fee drag and works in both bull and bear markets.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
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
    
    # Load daily data for Camarilla pivot calculation and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate pivot point and range
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    
    # Camarilla R3 and S3 levels (stronger levels)
    r3 = daily_pivot + daily_range * 1.1000
    s3 = daily_pivot - daily_range * 1.1000
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily levels and 1d EMA34 to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema1d_trend = ema34_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price closes above R3 with close > 1d EMA34 and volume > 1.5x 20MA
            if close[i] > r3_val and close[i] > ema1d_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S3 with close < 1d EMA34 and volume > 1.5x 20MA
            elif close[i] < s3_val and close[i] < ema1d_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1d EMA34 (trend reversal)
            if close[i] < ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 1d EMA34 (trend reversal)
            if close[i] > ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals