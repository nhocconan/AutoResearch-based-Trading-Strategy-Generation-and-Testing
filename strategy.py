#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: On 12-hour timeframe, trade Camarilla pivot level bounces with daily trend filter and volume confirmation.
Long when price rebounds from S3 level with daily EMA(50) trending up and volume > 1.5x 20-period average.
Short when price rebounds from R3 level with daily EMA(50) trending down and volume > 1.5x 20-period average.
Exit when price reaches opposite pivot level (S3/R3) or crosses the daily pivot (PP).
Uses actual Camarilla formula: H-L range from previous day, with S3/R3 as strong support/resistance.
Designed for 15-25 trades/year to minimize fee decay while capturing mean-reversion bounces in ranges and pullbacks in trends.
Works in both bull/bear markets as Camarilla adapts to volatility and daily filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
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
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Determine daily trend direction (using EMA slope)
    daily_trend_up = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    daily_trend_down = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    for i in range(1, len(ema_50_1d_aligned)):
        if not np.isnan(ema_50_1d_aligned[i]) and not np.isnan(ema_50_1d_aligned[i-1]):
            daily_trend_up[i] = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            daily_trend_down[i] = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
    
    # Calculate Camarilla levels from previous day's range
    # H, L, C from previous daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range from previous day (using close-to-close for stability)
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = close_1d[0]  # first bar uses its own close
    hl_range = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - prev_close), np.abs(low_1d - prev_close)))
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Camarilla levels
    # S3 = C - (H - L) * 1.1 / 2
    # S2 = C - (H - L) * 1.1 / 4
    # S1 = C - (H - L) * 1.1 / 6
    # R1 = C + (H - L) * 1.1 / 6
    # R2 = C + (H - L) * 1.1 / 4
    # R3 = C + (H - L) * 1.1 / 2
    s3 = close_1d - hl_range * 1.1 / 2.0
    r3 = close_1d + hl_range * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R3 or crosses above daily pivot
            if close[i] >= r3_aligned[i] or close[i] > pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S3 or crosses below daily pivot
            if close[i] <= s3_aligned[i] or close[i] < pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and daily trend alignment
            if vol_ok:
                # Long: price rebounds from S3 with daily uptrend
                if (close[i] > s3_aligned[i] and close[i-1] <= s3_aligned[i-1] and 
                    daily_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price rebounds from R3 with daily downtrend
                elif (close[i] < r3_aligned[i] and close[i-1] >= r3_aligned[i-1] and 
                      daily_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals