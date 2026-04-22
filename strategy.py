#!/usr/bin/env python3
"""
Hypothesis: 1D Donchian Channel breakout with weekly trend filter and volume confirmation.
In bull markets, price breaks above 20-day high with upward weekly trend.
In bear markets, price breaks below 20-day low with downward weekly trend.
Volume surge confirms institutional participation.
Designed for low trade frequency (10-25/year) to minimize fee drag on daily timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate Donchian Channel (20-period) on daily
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate weekly EMA34 trend
    ema34_weekly = pd.Series(df_weekly['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        
        if position == 0:
            # Long: Price breaks above 20-day high with bullish weekly trend and volume
            if (close[i] > donchian_high[i] and 
                close[i] > ema34_aligned[i] and  # Price above weekly EMA = bullish trend
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low with bearish weekly trend and volume
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema34_aligned[i] and  # Price below weekly EMA = bearish trend
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to Donchian midpoint
            if position == 1:
                if close[i] < donchian_mid[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_mid[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_DonchianBreakout_1wEMA34Trend_Volume"
timeframe = "1d"
leverage = 1.0