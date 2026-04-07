#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + Volume Confirmation + Daily Trend Filter
# Hypothesis: Breakout of 4h Donchian(20) with volume > 20-period average, 
# filtered by daily EMA(20) trend direction. Works in bull/bear by trading 
# with daily trend. Target: 100-200 total trades over 4 years (25-50/year).

name = "4h_donchian_breakout_vol_trend_v1"
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
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Daily EMA(20) for trend filter
    close_daily = df_daily['close'].values
    ema_20_daily = pd.Series(close_daily).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h = align_htf_to_ltf(prices, df_daily, ema_20_daily)
    
    # 4h Donchian(20) - upper and lower channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 4h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_20_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price touches lower Donchian or trend changes
            if low[i] <= donch_low[i] or close[i] < ema_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price touches upper Donchian or trend changes
            if high[i] >= donch_high[i] or close[i] > ema_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of daily trend with volume confirmation
            if vol_ok:
                if close[i] > ema_20_4h[i]:  # Uptrend - look for long breakout
                    if high[i] > donch_high[i]:
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend - look for short breakdown
                    if low[i] < donch_low[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals