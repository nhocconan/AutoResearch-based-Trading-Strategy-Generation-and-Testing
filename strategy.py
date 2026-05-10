#!/usr/bin/env python3
# 12H_Camarilla_R3_S3_Breakout_1dTrend_VolumeS
# Hypothesis: 12h Camarilla R3/S3 level breakouts with daily trend filter and volume confirmation.
# Camarilla levels provide high-probability reversal/breakout points. Daily trend ensures directional bias.
# Volume filter confirms breakout strength. Designed for low trade frequency (~15-30/year) to minimize fee drag.
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend via short signals).

name = "12H_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
timeframe = "12h"
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
    
    # Calculate Camarilla levels using previous day's OHLC
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # We need previous day's data, so we shift by 1
    daily_high = get_htf_data(prices, '1d')['high'].values
    daily_low = get_htf_data(prices, '1d')['low'].values
    daily_close = get_htf_data(prices, '1d')['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_R3 = daily_close + (daily_high - daily_low) * 1.1 / 2
    camarilla_S3 = daily_close - (daily_high - daily_low) * 1.1 / 2
    
    # Align to 12h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), camarilla_S3)
    
    # Daily trend filter: EMA 34
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 24-period average (24*12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Need enough history for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or \
           np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_uptrend = close[i] > ema_34_1d_aligned[i]
        is_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above Camarilla R3 + volume confirmation + daily uptrend
            if close[i] > camarilla_R3_aligned[i] and volume[i] > vol_threshold[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Camarilla S3 + volume confirmation + daily downtrend
            elif close[i] < camarilla_S3_aligned[i] and volume[i] > vol_threshold[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below Camarilla S3 or daily trend turns down
            if close[i] < camarilla_S3_aligned[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Camarilla R3 or daily trend turns up
            if close[i] > camarilla_R3_aligned[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals