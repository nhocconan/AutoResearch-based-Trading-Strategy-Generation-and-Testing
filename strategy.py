#!/usr/bin/env python3
# 4h_ema_breakout_volume_regime_v1
# Hypothesis: Uses 4H EMA20 breakout with 12H EMA50 trend filter and volume surge for entries.
# Long when: price > EMA20, 12H EMA50 slope up, volume > 1.5x avg.
# Short when: price < EMA20, 12H EMA50 slope down, volume > 1.5x avg.
# Exit when price crosses EMA20 opposite direction or volume drops below average.
# Designed for low trade frequency (<40/year) to avoid fee drag, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema_breakout_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4H EMA20 for entry/exit
    ema_period = 20
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Volume filter: 1.5x 20-period EMA (more responsive than SMA)
    vol_ema_period = 20
    vol_ema = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema = vol_series.ewm(span=vol_ema_period, adjust=False, min_periods=vol_ema_period).mean().values
    
    vol_surge = volume > 1.5 * vol_ema
    
    # Get 12H data for trend filter (EMA50 slope)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate slope: positive if current EMA > EMA 3 periods ago
    ema50_slope_12h = np.full(len(close_12h), np.nan)
    for i in range(3, len(close_12h)):
        if not np.isnan(ema50_12h[i]) and not np.isnan(ema50_12h[i-3]):
            ema50_slope_12h[i] = ema50_12h[i] - ema50_12h[i-3]
    ema50_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_slope_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(ema_period, vol_ema_period, 3) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20[i]) or np.isnan(vol_ema[i]) or np.isnan(ema50_slope_12h_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below EMA20 or volume drops below average
            if close[i] < ema20[i] or volume[i] < vol_ema[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above EMA20 or volume drops below average
            if close[i] > ema20[i] or volume[i] < vol_ema[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above EMA20, 12H EMA50 slope up, volume surge
            if (close[i] > ema20[i] and 
                ema50_slope_12h_aligned[i] > 0 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below EMA20, 12H EMA50 slope down, volume surge
            elif (close[i] < ema20[i] and 
                  ema50_slope_12h_aligned[i] < 0 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals