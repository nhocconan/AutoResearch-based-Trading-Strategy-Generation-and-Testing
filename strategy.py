#!/usr/bin/env python3
# 12h_1D_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Use daily Camarilla R3/S3 levels as support/resistance on 12h chart.
# Long when: price breaks above R3 with volume spike AND 1d EMA34 rising
# Short when: price breaks below S3 with volume spike AND 1d EMA34 falling
# Exit when price crosses back through R3/S3 OR 1d EMA34 trend reverses.
# Camarilla levels provide institutional reversal points; EMA34 filters counter-trend moves.
# Works in bull by buying breakouts in uptrend; works in bear by selling breakdowns in downtrend.
# Target: 15-30 trades/year (60-120 total over 4 years) to avoid fee drag.

name = "12h_1D_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla levels (R3, S3) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pp_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3 = PP + (H-L)*1.1/2, S3 = PP - (H-L)*1.1/2
    r3_1d = pp_1d + range_1d * 1.1 / 2
    s3_1d = pp_1d - range_1d * 1.1 / 2
    
    # --- 1d EMA34 trend ---
    ema_1d = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i < 34:
            ema_1d[i] = np.nan
        elif i == 34:
            ema_1d[i] = np.mean(close_1d[0:34])
        else:
            ema_1d[i] = (close_1d[i] * 2 / (34 + 1)) + (ema_1d[i-1] * (33 / (34 + 1)))
    
    # EMA slope
    ema_slope_1d = np.full(len(close_1d), np.nan)
    for i in range(35, len(close_1d)):
        ema_slope_1d[i] = ema_1d[i] - ema_1d[i-1]
    
    # --- 12h volume MA(20) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d indicators to 12h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(1d data needs 34 bars, volume MA20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > r3_1d_aligned[i]
        breakout_short = close[i] < s3_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if breakout_long and ema_slope_1d_aligned[i] > 0 and vol_spike:
                # Long: breakout above R3 in uptrend
                signals[i] = 0.25
                position = 1
            elif breakout_short and ema_slope_1d_aligned[i] < 0 and vol_spike:
                # Short: breakdown below S3 in downtrend
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price breaks below R3 OR EMA34 trend turns down
                if close[i] < r3_1d_aligned[i] or ema_slope_1d_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above S3 OR EMA34 trend turns up
                if close[i] > s3_1d_aligned[i] or ema_slope_1d_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals