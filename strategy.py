#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance.
Breakouts above R3 or below S3 with daily trend alignment and volume surge capture
institutional moves. Works in bull/bear by using daily trend filter to avoid counter-trend
breakouts in ranging markets. Target: 15-30 trades/year.
"""
name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # === Daily High/Low for Camarilla Calculation ===
    df_1d = get_htf_data(prices, '1d')
    if df_1d is None or len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close (must be completed day)
    prev_high = df_1d['high'].shift(1).values  # shift(1) for completed day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    range_val = prev_high - prev_low
    # Avoid division by zero
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    camarilla_r3 = prev_close + range_val * 1.1 / 4
    camarilla_s3 = prev_close - range_val * 1.1 / 4
    camarilla_r4 = prev_close + range_val * 1.1 / 2
    camarilla_s4 = prev_close - range_val * 1.1 / 2
    
    # Align to 6h timeframe (wait for daily bar to complete)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === Daily Trend Filter (EMA 34) ===
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume Spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Require strong volume surge
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with daily uptrend and volume spike
            if (close[i] > camarilla_r3_aligned[i] and
                close[i] > ema_34_aligned[i] and  # Price above daily EMA = uptrend
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with daily downtrend and volume spike
            elif (close[i] < camarilla_s3_aligned[i] and
                  close[i] < ema_34_aligned[i] and  # Price below daily EMA = downtrend
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price below S3 (failed breakout) or volume dies
            if close[i] < camarilla_s3_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above R3 (failed breakdown) or volume dies
            if close[i] > camarilla_r3_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals