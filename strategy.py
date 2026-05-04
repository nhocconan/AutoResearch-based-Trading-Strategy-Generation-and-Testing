#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA21 trend filter and volume spike confirmation
# Donchian channel (20-day high/low) captures medium-term trends. A breakout above the upper band
# or below the lower band with volume confirmation (1.5x 20-day volume EMA) and alignment with
# 1-week EMA21 trend provides high-probability entries. Designed for 1d timeframe to target
# 30-100 trades over 4 years (7-25/year) with discrete sizing (0.25). Works in bull markets
# by buying breakouts above upper band in uptrends and in bear markets by selling breakdowns
# below lower band in downtrends, avoiding false breakouts via trend and volume filters.

name = "1d_Donchian20_1wEMA21_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 1w EMA21 for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate Donchian(20) channels from 1d data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x 20-period EMA on volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: close breaks above upper Donchian + volume confirmation + price above 1w EMA21 (uptrend)
            if (close[i] > donchian_upper[i] and volume_confirmed and 
                close[i] > ema_21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below lower Donchian + volume confirmation + price below 1w EMA21 (downtrend)
            elif (close[i] < donchian_lower[i] and volume_confirmed and 
                  close[i] < ema_21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below lower Donchian (mean reversion) OR below 1w EMA21 (trend change)
            if close[i] < donchian_lower[i] or close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above upper Donchian (mean reversion) OR above 1w EMA21 (trend change)
            if close[i] > donchian_upper[i] or close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals