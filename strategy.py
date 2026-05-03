#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels provide precise intraday support/resistance derived from prior day's range.
# Breakouts above R1 or below S1 with volume confirmation capture institutional participation.
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws.
# Designed for 75-200 total trades over 4 years (19-50/year) with discrete position sizing.
# Works in bull markets via upward R1 breakouts and bear markets via downward S1 breakdowns.

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_R1 = np.full(n, np.nan)
    camarilla_S1 = np.full(n, np.nan)
    
    # Use prior 1d bar's OHLC to calculate levels for current 4h bar
    for i in range(len(df_1d)):
        # Get the timestamp of this 1d bar
        day_ts = df_1d.index[i]
        # Find all 4h bars that fall within this 1d period
        mask = (open_time >= day_ts) & (open_time < day_ts + pd.Timedelta(days=1))
        if mask.any():
            # Calculate Camarilla levels using this 1d bar's OHLC
            h = df_1d['high'].iloc[i]
            l = df_1d['low'].iloc[i]
            c = df_1d['close'].iloc[i]
            R1 = c + 1.1 * (h - l) / 12
            S1 = c - 1.1 * (h - l) / 12
            camarilla_R1[mask] = R1
            camarilla_S1[mask] = S1
    
    # Volume confirmation: 20-period EMA on 4h
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to ensure we have prior day data
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R1[i]) or np.isnan(camarilla_S1[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above Camarilla R1 in uptrend alignment with volume spike
            if close[i] > camarilla_R1[i] and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 in downtrend alignment with volume spike
            elif close[i] < camarilla_S1[i] and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S1 or loses uptrend alignment
            if close[i] < camarilla_S1[i] or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R1 or loses downtrend alignment
            if close[i] > camarilla_R1[i] or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals