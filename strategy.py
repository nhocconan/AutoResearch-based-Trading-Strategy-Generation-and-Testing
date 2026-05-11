#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_R3S3_Breakout_TrendVolume
Hypothesis: Price breaking above 4h R3 or below 4h S3 with daily trend confirmation (EMA34) and volume spike. Uses 4h pivot levels as strong support/resistance. In uptrend (price > EMA34 daily), buy breakouts above R3; in downtrend (price < EMA34 daily), sell breakdowns below S3. Volume confirms institutional interest. Designed for 1h timeframe with 4h pivot structure and daily trend filter to reduce trades and increase win rate. Works in both bull (breakouts) and bear (breakdowns) markets by capturing strong momentum moves after breaking key 4h levels.
"""

name = "1h_4h1d_Camarilla_R3S3_Breakout_TrendVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for pivot points
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 4h pivot points using previous 4h bar's OHLC
    h4_high = df_4h['high'].values
    h4_low = df_4h['low'].values
    h4_close = df_4h['close'].values
    
    # Shift to use previous 4h bar's data (avoid look-ahead)
    h4_high_prev = np.roll(h4_high, 1)
    h4_low_prev = np.roll(h4_low, 1)
    h4_close_prev = np.roll(h4_close, 1)
    # First period: use current values to avoid NaN
    h4_high_prev[0] = h4_high[0]
    h4_low_prev[0] = h4_low[0]
    h4_close_prev[0] = h4_close[0]
    
    # Calculate 4h pivot point
    h4_pivot = (h4_high_prev + h4_low_prev + h4_close_prev) / 3.0
    # Calculate 4h R3 and S3 levels
    h4_r3 = h4_close_prev + (1.1/4) * (h4_high_prev - h4_low_prev)
    h4_s3 = h4_close_prev - (1.1/4) * (h4_high_prev - h4_low_prev)
    
    # Align 4h R3/S3 to 1h timeframe
    h4_r3_aligned = align_htf_to_ltf(prices, df_4h, h4_r3)
    h4_s3_aligned = align_htf_to_ltf(prices, df_4h, h4_s3)
    h4_pivot_aligned = align_htf_to_ltf(prices, df_4h, h4_pivot)
    
    # Daily trend filter (EMA 34)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period average on 1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(h4_r3_aligned[i]) or np.isnan(h4_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            not in_session[i]):
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: break above 4h R3 + above daily EMA34 + volume spike
            if (close[i] > h4_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: break below 4h S3 + below daily EMA34 + volume spike
            elif (close[i] < h4_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: return to 4h pivot or trend reversal
            if position == 1:
                # Exit long: price returns to 4h pivot OR trend turns down
                if (close[i] <= h4_pivot_aligned[i]) or \
                   (close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price returns to 4h pivot OR trend turns up
                if (close[i] >= h4_pivot_aligned[i]) or \
                   (close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals