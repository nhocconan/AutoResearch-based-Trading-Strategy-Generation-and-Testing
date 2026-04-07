#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + 12h Trend + Volume Confirmation
# Hypothesis: Donchian breakouts capture momentum in both bull and bear markets.
# Use 12h EMA for trend filter to avoid counter-trend trades. Volume confirms institutional participation.
# 4h timeframe balances responsiveness and noise. Target: 20-50 trades/year (80-200 over 4 years).
# Works in bull markets by catching breakouts, in bear markets by avoiding false breaks via trend filter.
name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Donchian Channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max()
    lower_channel = low_series.rolling(window=20, min_periods=20).min()
    middle_channel = (upper_channel + lower_channel) / 2
    
    # 12-hour EMA(50) for trend filter
    daily_close_12h = df_12h['close'].values
    daily_ema_12h = pd.Series(daily_close_12h).ewm(span=50, adjust=False).mean().values
    daily_ema_4h = align_htf_to_ltf(prices, df_12h, daily_ema_12h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(daily_ema_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches middle channel or breaks below lower channel with volume
            if close[i] <= middle_channel[i] or (close[i] < lower_channel[i] and vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches middle channel or breaks above upper channel with volume
            if close[i] >= middle_channel[i] or (close[i] > upper_channel[i] and vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long: price breaks above upper channel with uptrend confirmation
                if close[i] > upper_channel[i] and close[i] > daily_ema_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower channel with downtrend confirmation
                elif close[i] < lower_channel[i] and close[i] < daily_ema_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals