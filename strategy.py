#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v1
Hypothesis: On 6-hour timeframe, use daily Camarilla pivot levels with EMA trend filter and volume confirmation.
Long when price breaks above R4 with daily EMA(50) trending up and volume > 1.5x 20-period average.
Short when price breaks below S4 with daily EMA(50) trending down and volume > 1.5x 20-period average.
Exit when price returns to the Pivot Point (PP) level.
Designed for 15-35 trades/year to minimize fee drag while capturing strong breakouts with institutional levels.
Works in both bull/bear markets as Camarilla levels adapt to volatility and daily trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels
    # PP = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4 = pp + (range_1d * 1.5)
    r3 = pp + (range_1d * 1.25)
    r2 = pp + (range_1d * 1.1666)
    r1 = pp + (range_1d * 1.0833)
    s1 = pp - (range_1d * 1.0833)
    s2 = pp - (range_1d * 1.1666)
    s3 = pp - (range_1d * 1.25)
    s4 = pp - (range_1d * 1.5)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Determine daily trend direction (using EMA slope)
    daily_trend_up = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    daily_trend_down = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    for i in range(1, len(ema_50_1d_aligned)):
        if not np.isnan(ema_50_1d_aligned[i]) and not np.isnan(ema_50_1d_aligned[i-1]):
            daily_trend_up[i] = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            daily_trend_down[i] = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
    
    # Align Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to or below Pivot Point
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above Pivot Point
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and daily trend alignment
            if vol_ok:
                # Long: price breaks above R4 with daily uptrend
                if (close[i] > r4_aligned[i] and close[i-1] <= r4_aligned[i-1] and 
                    daily_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below S4 with daily downtrend
                elif (close[i] < s4_aligned[i] and close[i-1] >= s4_aligned[i-1] and 
                      daily_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals