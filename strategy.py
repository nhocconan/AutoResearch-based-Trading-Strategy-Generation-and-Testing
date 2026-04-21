#!/usr/bin/env python3
"""
6h_Pivot_WW_Reversal_VolumeFilter
Hypothesis: 6h strategy trading weekly Camarilla pivot reversals filtered by 1d trend and volume spike.
Long when price < S3 and 1d EMA50 rising + volume > 2x average; short when price > R3 and 1d EMA50 falling + volume > 2x average.
Exits on opposite S1/R1 level or trend reversal. Designed to capture mean reversion in ranging markets and avoid false breakouts.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivot and EMA trend, 1w for weekly context)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation (based on previous 1d bar) ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    r3_1d = df_1d_close + 1.1 * range_1d
    s3_1d = df_1d_close - 1.1 * range_1d
    r4_1d = df_1d_close + 1.382 * range_1d
    s4_1d = df_1d_close - 1.382 * range_1d
    
    # Align 1d Camarilla levels to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_1d_prev = np.roll(ema_50_1d_aligned, 1)
    ema_50_1d_prev[0] = np.nan
    ema_rising = ema_50_1d_aligned > ema_50_1d_prev
    ema_falling = ema_50_1d_aligned < ema_50_1d_prev
    
    # === 6h ATR (14-period) for dynamic sizing ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (50-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i])
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 2.0x average (strict filter)
        volume_confirmed = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price below S3 (oversold) + rising 1d EMA50 + volume spike
            long_condition = (price < s3) and ema_rising[i] and volume_confirmed
            # Short: price above R3 (overbought) + falling 1d EMA50 + volume spike
            short_condition = (price > r3) and ema_falling[i] and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price > S1 (mean reversion) or trend turns bearish
            if price > s1 or not ema_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price < R1 (mean reversion) or trend turns bullish
            if price < r1 or not ema_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_WW_Reversal_VolumeFilter"
timeframe = "6h"
leverage = 1.0