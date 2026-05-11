#!/usr/bin/env python3
"""
12h_Pivot_R3_S3_Trend_Filter_v1
Hypothesis: Uses daily Camarilla pivot levels (R3/S3) as entry triggers on 12h timeframe, 
filtered by 1-day EMA34 trend direction and volume spike confirmation. 
Designed for low trade frequency (~15-25 trades/year) by requiring confluence of 
price touching R3/S3 levels, trend alignment, and volume confirmation. 
Works in both bull and bear markets by using trend filter to determine direction 
of breakout (long when above EMA34, short when below).
"""

name = "12h_Pivot_R3_S3_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Daily EMA34 for trend filter ---
    close_1d = df_1d['close']
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Daily Camarilla Pivot Levels (R3, S3) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Calculate pivot point
    pp = (high_1d + low_1d + close_1d_arr) / 3.0
    # Calculate ranges
    range_hl = high_1d - low_1d
    # Camarilla levels
    r3 = pp + range_hl * 1.1 / 2.0  # R3 = PP + (H-L)*1.1/2
    s3 = pp - range_hl * 1.1 / 2.0  # S3 = PP - (H-L)*1.1/2
    
    # Align pivot levels to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # --- Volume Spike Detection (2.0x 20-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ema.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend based on price vs daily EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for R3/S3 touch with volume confirmation and trend alignment
            # Long when price touches/runs above R3 in uptrend
            if price_above_ema and high[i] >= r3_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short when price touches/runs below S3 in downtrend
            elif price_below_ema and low[i] <= s3_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price falls back below pivot point or trend reverses
                pp_aligned = ((r3_aligned[i] + s3_aligned[i]) / 2.0)  # PP = (R3+S3)/2
                exit_signal = (close[i] < pp_aligned) or (close[i] < ema_34_1d_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises back above pivot point or trend reverses
                pp_aligned = ((r3_aligned[i] + s3_aligned[i]) / 2.0)  # PP = (R3+S3)/2
                exit_signal = (close[i] > pp_aligned) or (close[i] > ema_34_1d_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals