#!/usr/bin/env python3
"""
1h_volume_breakout_4h1d_trend_v1
Hypothesis: On 1-hour timeframe, use volume breakouts aligned with 4h and 1d trend filters to reduce false signals.
Long when price breaks above 20-period high with volume > 2x average, 4h EMA(20) rising, and 1d close > EMA(50).
Short when price breaks below 20-period low with volume > 2x average, 4h EMA(20) falling, and 1d close < EMA(50).
Exit when price returns to 20-period midpoint. Uses session filter (08-20 UTC) to avoid low-volume hours.
Designed for 15-35 trades/year to minimize fee shock while capturing strong intraday moves with multi-timeframe confirmation.
Works in bull/bear markets as trend filters prevent counter-trend trades and volume breaks signal institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_breakout_4h1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period high/low and midpoint
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    high_20 = high_series.rolling(window=20, min_periods=20).max().values
    low_20 = low_series.rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 30, 50), n):
        # Skip if data not available or outside session
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or not session_mask[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: at least 2x average
        vol_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to 20-period midpoint
            if close[i] <= mid_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price returns to 20-period midpoint
            if close[i] >= mid_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Only enter with volume confirmation and multi-timeframe trend alignment
            if vol_ok:
                # Long: price breaks above 20-period high with 4h uptrend and 1d uptrend
                if (close[i] > high_20[i] and close[i-1] <= high_20[i-1] and 
                    ema_20_4h_aligned[i] > ema_20_4h_aligned[i-1] and
                    close[i] > ema_50_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.20
                # Short: price breaks below 20-period low with 4h downtrend and 1d downtrend
                elif (close[i] < low_20[i] and close[i-1] >= low_20[i-1] and 
                      ema_20_4h_aligned[i] < ema_20_4h_aligned[i-1] and
                      close[i] < ema_50_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.20
    
    return signals