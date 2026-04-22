#!/usr/bin/env python3
"""
Hypothesis: 4-hour Chaikin Money Flow (CMF) with 1-day trend filter and volume confirmation.
Long when CMF > 0.1 and 1-day EMA(34) trend is up and 1-day volume > 50-day average volume.
Short when CMF < -0.1 and 1-day EMA(34) trend is down and 1-day volume > 50-day average volume.
Exit when CMF crosses zero or volume filter fails.
CMF measures institutional money flow, EMA trend filters for direction, volume confirms participation.
Designed for 4h timeframe to target 20-50 trades/year per symbol with strong edge in both bull and bear markets.
"""

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
    
    # Load 1-day data for trend and volume filters - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-day EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_prev = np.roll(ema_34_1d, 1)
    ema_34_1d_prev[0] = ema_34_1d[0]  # avoid NaN
    ema_trend_up = ema_34_1d > ema_34_1d_prev
    ema_trend_down = ema_34_1d < ema_34_1d_prev
    ema_trend_up_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_up)
    ema_trend_down_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_down)
    
    # 1-day volume and its 50-day average for volume filter
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    volume_filter = volume_1d > avg_vol_1d_aligned
    
    # Chaikin Money Flow (CMF) calculation for 4h period
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    mfm = np.zeros_like(close)
    hl_range = high - low
    # Avoid division by zero
    valid_range = hl_range != 0
    mfm[valid_range] = ((close[valid_range] - low[valid_range]) - (high[valid_range] - close[valid_range])) / hl_range[valid_range]
    # Money Flow Volume = Money Flow Multiplier * Volume
    mfv = mfm * volume
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.zeros_like(close)
    vol_sum_valid = vol_sum != 0
    cmf[vol_sum_valid] = mfv_sum[vol_sum_valid] / vol_sum[vol_sum_valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(cmf[i]) or np.isnan(ema_trend_up_aligned[i]) or np.isnan(ema_trend_down_aligned[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CMF > 0.1, uptrend, and volume confirmation
            if cmf[i] > 0.1 and ema_trend_up_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.1, downtrend, and volume confirmation
            elif cmf[i] < -0.1 and ema_trend_down_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: CMF crosses below zero or volume filter fails
                if cmf[i] < 0 or not volume_filter[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: CMF crosses above zero or volume filter fails
                if cmf[i] > 0 or not volume_filter[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_CMF_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0