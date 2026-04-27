#!/usr/bin/env python3
"""
12h_Camarilla_R4_S4_Breakout_1wTrend_VolumeSpike
Hypothesis: Uses weekly Camarilla pivot levels (R4/S4) for breakout entries in the direction of 1w trend (price > EMA50). Volume confirmation (>2x average) ensures conviction. Exits when price reverts to R3/S3 or trend reverses. 12h timeframe targets 50-150 trades over 4 years (12-37/year). Works in bull markets via upside breakouts and in bear markets via downside breakdowns. Camarilla R4/S4 represent extreme weekly levels where breakouts often continue, while R3/S3 act as profit-taking zones. Weekly timeframe provides more stable trends and filters noise, suitable for BTC/ETH in choppy/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w Camarilla pivot levels
    # Based on previous week's high, low, close
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla calculations use previous week's data
    # We need to shift by 1 to use previous week's HLC for current week's levels
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    # First bar has no previous week
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    # Camarilla levels
    # R4 = Close + (High - Low) * 1.1 / 2
    # S4 = Close - (High - Low) * 1.1 / 2
    # R3 = Close + (High - Low) * 1.1 / 4
    # S3 = Close - (High - Low) * 1.1 / 4
    range_1w = prev_high_1w - prev_low_1w
    camarilla_r4 = prev_close_1w + range_1w * 1.1 / 2
    camarilla_s4 = prev_close_1w - range_1w * 1.1 / 2
    camarilla_r3 = prev_close_1w + range_1w * 1.1 / 4
    camarilla_s3 = prev_close_1w - range_1w * 1.1 / 4
    
    # Align all 1w indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need EMA50 (50), volume avg (20), and Camarilla (need previous week)
    start_idx = max(50, 20, 2)  # +2 for Camarilla shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1w_val = ema_50_1w_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend: price > EMA50 = uptrend, price < EMA50 = downtrend
            is_uptrend = close_val > ema_1w_val
            is_downtrend = close_val < ema_1w_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above R4 and volume confirms
                if (close_val > r4) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below S4 and volume confirms
                if (close_val < s4) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price reverts to R3 or trend changes to downtrend
            exit_condition = (close_val < r3) or (close_val < ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reverts to S3 or trend changes to uptrend
            exit_condition = (close_val > s3) or (close_val > ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R4_S4_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0