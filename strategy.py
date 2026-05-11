#!/usr/bin/env python3
"""
4H_Camarilla_R3_S3_Breakout_12hTrend_Volume
Hypothesis: Breakouts from daily Camarilla R3/S3 levels with 12h trend filter and volume confirmation.
In strong trends, price breaking above R3 or below S3 with volume confirms institutional participation.
Trades only in direction of higher timeframe trend to avoid counter-trend whipsaws.
Designed for low frequency (15-35 trades/year) to minimize fee drag in both bull and bear markets.
"""

name = "4H_Camarilla_R3_S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla pivot levels (R3, S3) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels: R3/S3
    r3 = pp + (high_1d - low_1d) * 1.1 / 4.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align R3/S3 to 4h timeframe (using previous day's levels for breakout)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # --- 12h EMA50 for trend filter ---
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- Volume Spike (4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: Breakout of R3/S3 with volume and trend alignment
        long_entry = (close[i] > r3_aligned[i]) and vol_spike[i] and (close[i] > ema_50_12h_aligned[i])
        short_entry = (close[i] < s3_aligned[i]) and vol_spike[i] and (close[i] < ema_50_12h_aligned[i])
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: Price returns to pivot level or trend reversal
            if position == 1:
                # Exit if price crosses below pivot or trend turns down
                pp_aligned = align_htf_to_ltf(prices, df_1d, (high_1d + low_1d + close_1d) / 3.0)
                if (close[i] < pp_aligned[i]) or (close[i] < ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit if price crosses above pivot or trend turns up
                pp_aligned = align_htf_to_ltf(prices, df_1d, (high_1d + low_1d + close_1d) / 3.0)
                if (close[i] > pp_aligned[i]) or (close[i] > ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals