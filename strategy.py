#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R (14) extreme reversal with 1d EMA50 trend filter and volume confirmation (>1.5x 20 EMA volume)
# Williams %R identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false signals (>1.5x average volume) - balanced to reduce trades
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "4h_WilliamsR14_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) trend filter from prior completed 1d bar
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_shifted = np.roll(ema_50_1d, 1)
    ema_50_1d_shifted[0] = np.nan
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Williams %R(14) from prior completed 4h bar
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    highest_high = high_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_series.rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    rr = highest_high - lowest_low
    williams_r = np.where(rr != 0, (highest_high - close_series) / rr * -100, -50)
    
    # Shift by 1 to use only prior completed 4h bar (no look-ahead)
    williams_r_shifted = np.roll(williams_r, 1)
    williams_r_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(williams_r_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume spike
            if williams_r_shifted[i] < -80 and close[i] > ema_50_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume spike
            elif williams_r_shifted[i] > -20 and close[i] < ema_50_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (momentum fading) OR price crosses below 1d EMA50
            if williams_r_shifted[i] > -50 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (momentum fading) OR price crosses above 1d EMA50
            if williams_r_shifted[i] < -50 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals