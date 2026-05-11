# 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT
# Hypothesis: Price breaking above daily R3 or below daily S3 with 12h trend confirmation (EMA50) and volume spike. Uses daily pivot levels as strong support/resistance. In uptrend (price > EMA50 on 12h), buy breakouts above R3; in downtrend (price < EMA50 on 12h), sell breakdowns below S3. Volume confirms institutional interest. Designed for 6h timeframe with daily pivot structure and 12h trend filter to reduce trades and increase win rate. Works in both bull (breakouts) and bear (breakdowns) markets by capturing strong momentum moves after breaking key daily levels.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate daily pivot points using previous day's OHLC
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Shift to use previous day's data (avoid look-ahead)
    d_high_prev = np.roll(d_high, 1)
    d_low_prev = np.roll(d_low, 1)
    d_close_prev = np.roll(d_close, 1)
    # First period: use current values to avoid NaN
    d_high_prev[0] = d_high[0]
    d_low_prev[0] = d_low[0]
    d_close_prev[0] = d_close[0]
    
    # Calculate daily pivot point
    d_pivot = (d_high_prev + d_low_prev + d_close_prev) / 3.0
    # Calculate daily R3 and S3 levels
    d_r3 = d_close_prev + (1.1/4) * (d_high_prev - d_low_prev)
    d_s3 = d_close_prev - (1.1/4) * (d_high_prev - d_low_prev)
    
    # Align daily R3/S3 to 6h timeframe
    d_r3_aligned = align_htf_to_ltf(prices, df_1d, d_r3)
    d_s3_aligned = align_htf_to_ltf(prices, df_1d, d_s3)
    d_pivot_aligned = align_htf_to_ltf(prices, df_1d, d_pivot)
    
    # 12h trend filter (EMA 50)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(d_r3_aligned[i]) or np.isnan(d_s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: break above daily R3 + above 12h EMA50 + volume spike
            if (close[i] > d_r3_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below daily S3 + below 12h EMA50 + volume spike
            elif (close[i] < d_s3_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to daily pivot or trend reversal
            if position == 1:
                # Exit long: price returns to daily pivot OR trend turns down
                if (close[i] <= d_pivot_aligned[i]) or \
                   (close[i] < ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to daily pivot OR trend turns up
                if (close[i] >= d_pivot_aligned[i]) or \
                   (close[i] > ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3