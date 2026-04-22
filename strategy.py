#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with 1-week trend filter and volume confirmation.
In bull markets, price breaks above 20-period Donchian high with upward 1-week trend.
In bear markets, price breaks below 20-period Donchian low with downward 1-week trend.
Volume surge confirms institutional participation. Designed for low trade frequency (12-37/year).
Uses 1-week EMA for trend filter and volume spike for confirmation.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1-week EMA34 trend
    ema34_weekly = pd.Series(df_weekly['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with bullish 1w trend and volume
            if (close[i] > donchian_high[i] and 
                close[i] > ema34_aligned[i] and  # Price above 1w EMA = bullish trend
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with bearish 1w trend and volume
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema34_aligned[i] and  # Price below 1w EMA = bearish trend
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to Donchian midpoint
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if position == 1:
                if close[i] < donchian_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_DonchianBreakout_1wEMA34Trend_Volume"
timeframe = "12h"
leverage = 1.0
#%%