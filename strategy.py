#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Alpha
# Hypothesis: Use 1d Camarilla pivot levels (R3/S3) for breakout signals with 1d EMA trend filter and volume spike confirmation.
# Long when price breaks above R3 with price > 1d EMA and volume > 2x MA; short when price breaks below S3 with price < 1d EMA and volume > 2x MA.
# Exit when price reverses back to the 1d EMA level. Designed to capture strong intraday moves with trend and volume filters.
# Targets 15-30 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Alpha"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on prior 1d bar's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d bar's OHLC for pivot calculation
    phigh = df_1d['high'].shift(1).values  # Previous 1d high
    plow = df_1d['low'].shift(1).values    # Previous 1d low
    pclose = df_1d['close'].shift(1).values # Previous 1d close
    
    # Camarilla calculations: R3/S3 levels
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    rng = phigh - plow
    r3 = pclose + rng * 1.1 / 4.0
    s3 = pclose - rng * 1.1 / 4.0
    
    # Align R3/S3 to 4h timeframe (wait for 1d bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA for trend filter (34-period)
    pclose_series = pd.Series(pclose)
    ema1d = pclose_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    # Volume confirmation: 20-period moving average on 4h data
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with price > 1d EMA and volume > 2x MA
            if close[i] > r3_aligned[i] and close[i] > ema1d_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with price < 1d EMA and volume > 2x MA
            elif close[i] < s3_aligned[i] and close[i] < ema1d_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves back below 1d EMA (trend invalidated)
            if close[i] < ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves back above 1d EMA (trend invalidated)
            if close[i] > ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals