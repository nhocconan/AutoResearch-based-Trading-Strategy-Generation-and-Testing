#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Spike_v1
Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance.
Breakouts above R3 or below S3 with volume spike and 1d EMA34 trend filter capture
institutional moves. Works in bull/bear by following 1d trend direction.
Target: 20-40 trades/year on 4h with volume confirmation reducing false breakouts.
"""

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Spike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (R3, S3) for each day
    # R3 = close + 1.1 * (high - low) / 1.25
    # S3 = close - 1.1 * (high - low) / 1.25
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 1.25
    s3 = prev_close - 1.1 * camarilla_range / 1.25
    
    # Align Camarilla levels to 4h (wait for previous day's close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detector (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA and volume)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1d EMA34
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Look for breakouts with volume spike
            if uptrend and vol_spike[i] and close[i] > r3_aligned[i]:
                # Break above R3 in uptrend -> long
                signals[i] = 0.25
                position = 1
            elif downtrend and vol_spike[i] and close[i] < s3_aligned[i]:
                # Break below S3 in downtrend -> short
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Camarilla body or trend reversal
            if position == 1:
                # Exit long: price back below R3 or trend turns down
                exit_signal = (close[i] < r3_aligned[i]) or (not uptrend)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price back above S3 or trend turns up
                exit_signal = (close[i] > s3_aligned[i]) or (not downtrend)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals