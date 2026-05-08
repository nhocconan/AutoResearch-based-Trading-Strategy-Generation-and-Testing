#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band squeeze + 4h trend + 1d volume confirmation
# Bollinger Band squeeze identifies low volatility periods that precede breakouts.
# 4h EMA trend filter ensures we only trade breakouts in the direction of the medium-term trend.
# 1d volume spike confirms institutional participation in the breakout.
# Targets 15-30 trades per year (~60-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by filtering for trend-aligned breakouts only.

name = "1h_BBSqueeze_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 1h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma + (bb_std * bb_std_dev)
    bb_lower = sma - (bb_std * bb_std_dev)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band squeeze: width below 20-period mean
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma
    
    # Get 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_period = 50
    ema_4h = pd.Series(close_4h).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma_1d * 2.0)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: BB squeeze breakout up + 4h uptrend + volume spike + session
            if (bb_squeeze[i] and 
                close[i] > bb_upper[i] and 
                close[i] > ema_4h_aligned[i] and 
                vol_spike_1d_aligned[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: BB squeeze breakout down + 4h downtrend + volume spike + session
            elif (bb_squeeze[i] and 
                  close[i] < bb_lower[i] and 
                  close[i] < ema_4h_aligned[i] and 
                  vol_spike_1d_aligned[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to SMA or 4h trend turns down
            if close[i] < sma[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to SMA or 4h trend turns up
            if close[i] > sma[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals