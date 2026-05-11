#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Breakout_1dTrend_Volume_v1
Hypothesis: Camarilla pivot levels (R3/S3) from 1-day timeframe act as significant support/resistance.
Breakouts above R3 or below S3 with volume confirmation and 1-day EMA trend filter capture strong moves.
Works in bull markets (breakouts above R3) and bear markets (breakdowns below S3).
Target: 15-35 trades per year on 4h timeframe (~60-140 total over 4 years).
"""

name = "4h_Camarilla_Pivot_Breakout_1dTrend_Volume_v1"
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
    
    # === 1D Data for Camarilla Pivots and Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses same day's high
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla pivot levels
    range_ = prev_high - prev_low
    camarilla_r3 = prev_close + range_ * 1.1 / 2
    camarilla_s3 = prev_close - range_ * 1.1 / 2
    
    # 1-day EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below EMA34
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: breakout above R3 with volume and uptrend
            if close[i] > r3_aligned[i] and vol_confirm and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S3 with volume and downtrend
            elif close[i] < s3_aligned[i] and vol_confirm and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S3 or trend reverses
            if close[i] < s3_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above R3 or trend reverses
            if close[i] > r3_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals