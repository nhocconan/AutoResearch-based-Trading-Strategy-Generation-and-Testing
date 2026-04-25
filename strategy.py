#!/usr/bin/env python3
"""
6h Camarilla Pivot Breakout + 12h EMA34 Trend Filter + Volume Spike Confirmation
Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) from 12h timeframe provide institutional support/resistance.
Breakouts above R4 or below S4 with volume confirmation and aligned with 12h EMA34 trend yield high-probability trades.
In ranging markets (price between R3/S3), we fade extremes at R3/S3 with volume confirmation.
Uses 6h primary timeframe with 12h Camarilla pivots and EMA34 for trend filter.
Designed for BTC/ETH with 50-150 total trades over 4 years to minimize fee drag while capturing institutional levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots and EMA34 (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need 34 for EMA34 + enough for pivots
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 12h
    # Based on previous 12h bar's high, low, close
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_12h + low_12h + close_12h) / 3.0
    
    # Calculate ranges
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r4 = pp + range_12h * 1.1 / 2.0
    r3 = pp + range_12h * 1.1 / 4.0
    s3 = pp - range_12h * 1.1 / 4.0
    s4 = pp - range_12h * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_12h, r4)
    r3_6h = align_htf_to_ltf(prices, df_12h, r3)
    s3_6h = align_htf_to_ltf(prices, df_12h, s3)
    s4_6h = align_htf_to_ltf(prices, df_12h, s4)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for pivots, EMA34, and volume MA
    start_idx = max(34, 20)  # 34 for EMA34/pivots, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        r4_val = r4_6h[i]
        r3_val = r3_6h[i]
        s3_val = s3_6h[i]
        s4_val = s4_6h[i]
        ema_34_val = ema_34_12h_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma
        
        # Trend direction from 12h EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        if position == 0:
            # Breakout continuation: price breaks R4/S4 with volume and trend alignment
            if curr_high > r4_val and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            elif curr_low < s4_val and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
            # Mean reversion fade: price reaches R3/S3 with volume, counter to trend
            elif curr_high >= r3_val and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
            elif curr_low <= s3_val and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below R3 OR breaks below S4 (stop)
            if curr_close < r3_val or curr_low < s4_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above S3 OR breaks above R4 (stop)
            if curr_close > s3_val or curr_high > r4_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_Pivot_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0