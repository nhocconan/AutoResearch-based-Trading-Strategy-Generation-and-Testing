#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA34 trend filter + 1d volume spike
# Williams %R identifies overbought/oversold conditions for mean reversion in ranging markets.
# In strong trends (ADX > 25), we fade extreme %R readings only when aligned with 1d EMA34 trend.
# Volume spike confirms institutional participation. Designed for 6h timeframe to capture
# swing reversals in both bull and bear markets with controlled trade frequency.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_trend_up = ema_34 > np.roll(ema_34, 1)  # Today's EMA > yesterday's EMA
    ema_trend_down = ema_34 < np.roll(ema_34, 1)  # Today's EMA < yesterday's EMA
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1d indicators to 6h timeframe
    ema_trend_up_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_up)
    ema_trend_down_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_down)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50.0, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_trend_up_aligned[i]) or 
            np.isnan(ema_trend_down_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + 1d EMA uptrend + volume spike
            if williams_r[i] < -80 and ema_trend_up_aligned[i] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + 1d EMA downtrend + volume spike
            elif williams_r[i] > -20 and ema_trend_down_aligned[i] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (exit oversold) OR reverse signal
            if williams_r[i] > -50 or (williams_r[i] > -20 and ema_trend_down_aligned[i] and volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (exit overbought) OR reverse signal
            if williams_r[i] < -50 or (williams_r[i] < -80 and ema_trend_up_aligned[i] and volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals