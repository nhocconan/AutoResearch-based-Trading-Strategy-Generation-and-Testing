#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA trend and Camarilla levels.
- Camarilla pivot levels: R3/S3 for breakout continuation, R4/S4 for extreme reversal fade.
- Trend filter: Only trade breakouts in direction of 1d EMA34 (long if price > EMA34, short if price < EMA34).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe.
- Works in bull via buying R3 breakouts in uptrend, in bear via selling S3 breakdowns in downtrend.
- Uses Camarilla structure which adapts to volatility and has proven edge in crypto.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels
    # Camarilla: based on previous day's range
    prev_close = close_1d[:-1]  # yesterday's close
    prev_high = high_1d[:-1]    # yesterday's high
    prev_low = low_1d[:-1]      # yesterday's low
    
    # True range components for Camarilla
    range_1d = prev_high - prev_low
    
    # Camarilla levels (based on previous day)
    camarilla_r4 = prev_close + range_1d * 1.500
    camarilla_r3 = prev_close + range_1d * 1.250
    camarilla_s3 = prev_close - range_1d * 1.250
    camarilla_s4 = prev_close - range_1d * 1.500
    
    # Align Camarilla levels to 6h (1 day = 4 * 6h bars)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4, additional_delay_bars=0)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3, additional_delay_bars=0)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3, additional_delay_bars=0)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4, additional_delay_bars=0)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20) + 1  # EMA34 + volume MA + 1 for prev day shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1d EMA34 trend
            if close[i] > ema_34_1d_aligned[i]:  # Uptrend
                # Breakout above R3: continuation long
                if close[i] > camarilla_r3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
            elif close[i] < ema_34_1d_aligned[i]:  # Downtrend
                # Breakdown below S3: continuation short
                if close[i] < camarilla_s3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below R3 or opposite signal
            if close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above S3 or opposite signal
            if close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0