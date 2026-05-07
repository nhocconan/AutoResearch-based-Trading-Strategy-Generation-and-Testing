#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla R3/S3 breakouts on 12h capture strong momentum with institutional participation.
# Confirmed by 1d EMA34 trend filter and volume spike (>2x average).
# Works in bull markets via long breakouts at R3 and bear via short breakdowns at S3.
# Volume filter reduces false breakouts, trend filter avoids counter-trend trades.
# Target: 12-37 trades per year (~50-150 over 4 years) with position size 0.25.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels on 12h using previous day's OHLC
    # We'll use rolling window of 288 bars (24h * 12 intervals per hour) to approximate daily
    lookback = 288
    if n < lookback:
        return np.zeros(n)
    
    # Calculate Camarilla for each bar using prior 288 bars' OHLC
    high_max = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    low_min = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    close_prev = pd.Series(close).shift(1).rolling(window=lookback, min_periods=lookback).last().values
    
    range_val = high_max - low_min
    # Avoid division by zero
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    # Camarilla levels
    R3 = close_prev + 1.1 * range_val * 1.1 / 12  # Actually: C + (H-L)*1.1/12
    S3 = close_prev - 1.1 * range_val * 1.1 / 12  # Actually: C - (H-L)*1.1/12
    
    # Volume ratio: current volume / 24-period average volume (2h average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback  # Need lookback period for Camarilla calculation
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(R3[i]) or np.isnan(S3[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks above R3 or below S3
        breakout_up = close[i] > R3[i-1]  # Use previous bar's R3 to avoid look-ahead
        breakout_down = close[i] < S3[i-1]  # Use previous bar's S3
        
        # Volume confirmation: volume > 2x average
        volume_confirm = vol_ratio[i] > 2.0
        
        # Trend filter from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: upward breakout at R3 + volume + uptrend
            if breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout at S3 + volume + downtrend
            elif breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks back below S3 or trend reversal
            if close[i] < S3[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks back above R3 or trend reversal
            if close[i] > R3[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals