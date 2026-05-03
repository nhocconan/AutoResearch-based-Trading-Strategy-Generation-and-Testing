#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels act as intraday support/resistance where price often reverses or accelerates.
# Breakout above R3 or below S3 with 1d EMA34 trend alignment captures strong moves.
# Volume confirmation ensures conviction. Designed for 12-30 trades/year on 12h to minimize fee drag.
# Works in both bull and bear markets by trading breakouts in the direction of the higher timeframe trend.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous day for Camarilla calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels using previous day's OHLC
        if i >= 24:  # Need at least 24 hours of data (1 day) for previous day OHLC
            # Get previous day's OHLC (24 bars back in 12h timeframe)
            prev_high = np.max(high[i-24:i])
            prev_low = np.min(low[i-24:i])
            prev_close = close[i-24]
            
            # Calculate Camarilla levels
            range_val = prev_high - prev_low
            if range_val > 0:
                camarilla_r3 = prev_close + range_val * 1.1 / 4
                camarilla_s3 = prev_close - range_val * 1.1 / 4
            else:
                camarilla_r3 = prev_high
                camarilla_s3 = prev_low
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA
        if i >= 19:
            vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Camarilla breakout conditions
        breakout_above_r3 = close[i] > camarilla_r3
        breakout_below_s3 = close[i] < camarilla_s3
        
        if position == 0:
            # Long: breakout above R3 in 1d uptrend with volume spike
            if breakout_above_r3 and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 in 1d downtrend with volume spike
            elif breakout_below_s3 and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 or loses 1d uptrend
            if close[i] < camarilla_r3 or ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 or loses 1d downtrend
            if close[i] > camarilla_s3 or ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals