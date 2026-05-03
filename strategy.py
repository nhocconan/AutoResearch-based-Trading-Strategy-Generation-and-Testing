#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide intraday support/resistance that work well in ranging and trending markets.
# R3 and S3 levels act as strong breakout points when price moves beyond the normal trading range.
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades.
# Volume confirmation filters false breakouts. Designed for 75-200 total trades over 4 years (19-50/year).
# Works in bull markets via upward breaks at R3 and in bear markets via downward breaks at S3.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla formula: 
    # H4 = close + 1.1*(high-low)/2
    # L4 = close - 1.1*(high-low)/2
    # R3 = close + 1.1*(high-low)/2
    # S3 = close - 1.1*(high-low)/2
    # Actually R3=H4, S3=L4
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    camarilla_high = close_series + 1.1 * (high_series - low_series) / 2
    camarilla_low = close_series - 1.1 * (high_series - low_series) / 2
    camarilla_high_prev = camarilla_high.shift(1).values
    camarilla_low_prev = camarilla_low.shift(1).values
    
    # Volume confirmation: 20-period EMA on 4h
    vol_ema_20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_20_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20[:] = vol_ema_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid Camarilla and volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_high_prev[i]) or np.isnan(camarilla_low_prev[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above camarilla high (R3/H4) in uptrend alignment with volume spike
            if close[i] > camarilla_high_prev[i] and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below camarilla low (S3/L4) in downtrend alignment with volume spike
            elif close[i] < camarilla_low_prev[i] and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below camarilla low or loses uptrend alignment
            if close[i] < camarilla_low_prev[i] or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above camarilla high or loses downtrend alignment
            if close[i] > camarilla_high_prev[i] or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals