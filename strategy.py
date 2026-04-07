#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Donchian Breakout with Volume and Trend Filter
# Hypothesis: Breakouts above/below Donchian(20) channels on 4h with volume confirmation
# and trend filter (price vs 200 EMA) work in both bull and bear markets.
# In bull markets: buy breakouts above upper band, sell breakouts below lower band.
# In bear markets: sell breakouts below lower band, buy breakouts above upper band.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_daily_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channel calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channel (20-period)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Calculate upper and lower bands
    high_series = pd.Series(daily_high)
    low_series = pd.Series(daily_low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    upper = np.roll(upper, 1)
    lower = np.roll(lower, 1)
    
    # Handle first element
    if len(upper) > 1:
        upper[0] = upper[1]
        lower[0] = lower[1]
    else:
        upper[0] = 0
        lower[0] = 0
    
    # Align to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_daily, upper)
    lower_aligned = align_htf_to_ltf(prices, df_daily, lower)
    
    # Trend filter: price vs 200 EMA
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume filter: volume > 1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(ema_200[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: breakout below lower band or trend reversal
            if (low[i] <= lower_aligned[i]) or (close[i] < ema_200[i]) or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit conditions: breakout above upper band or trend reversal
            if (high[i] >= upper_aligned[i]) or (close[i] > ema_200[i]) or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above upper band with volume and trend
            if (high[i] > upper_aligned[i]) and close[i] > ema_200[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: breakout below lower band with volume and trend
            elif (low[i] < lower_aligned[i]) and close[i] < ema_200[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals