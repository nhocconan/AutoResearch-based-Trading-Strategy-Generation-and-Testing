#!/usr/bin/env python3
"""
12h_aroon_1d_trend_volume_v1
Hypothesis: On 12-hour timeframe, use Aroon oscillator (trend strength) with 1-day EMA trend filter and volume confirmation.
Long when Aroon Up > 70 and Aroon Down < 30 (strong uptrend) with daily EMA(50) rising and volume > 1.5x 20-period average.
Short when Aroon Down > 70 and Aroon Up < 30 (strong downtrend) with daily EMA(50) falling and volume > 1.5x 20-period average.
Exit when Aroon oscillator indicates weakening trend (|Aroon Up - Aroon Down| < 20).
Designed for 15-30 trades/year to minimize fee drift while capturing sustained trends.
Aroon adapts to volatility and daily trend filter prevents counter-trend trades in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_aroon_1d_trend_volume_v1"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
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
    
    # Calculate Aroon Oscillator (25-period) on 12h timeframe
    aroon_period = 25
    # Days since highest high
    high_series = pd.Series(high)
    high_roll_idx = high_series.rolling(window=aroon_period, min_periods=aroon_period).apply(
        lambda x: aroon_period - 1 - np.argmax(x), raw=True
    )
    aroon_up = ((aroon_period - high_roll_idx) / aroon_period) * 100
    
    # Days since lowest low
    low_series = pd.Series(low)
    low_roll_idx = low_series.rolling(window=aroon_period, min_periods=aroon_period).apply(
        lambda x: aroon_period - 1 - np.argmin(x), raw=True
    )
    aroon_down = ((aroon_period - low_roll_idx) / aroon_period) * 100
    
    # Handle NaN values from insufficient data
    aroon_up = aroon_up.values
    aroon_down = aroon_down.values
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(25, 50), n):
        # Skip if data not available
        if (np.isnan(aroon_up[i]) or np.isnan(aroon_down[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: trend weakening (Aroon spread < 20)
            if abs(aroon_up[i] - aroon_down[i]) < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend weakening (Aroon spread < 20)
            if abs(aroon_up[i] - aroon_down[i]) < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and daily trend alignment
            if vol_ok:
                # Long: strong uptrend (Aroon Up > 70, Aroon Down < 30) with daily uptrend
                if (aroon_up[i] > 70 and aroon_down[i] < 30 and 
                    daily_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: strong downtrend (Aroon Down > 70, Aroon Up < 30) with daily downtrend
                elif (aroon_down[i] > 70 and aroon_up[i] < 30 and 
                      daily_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals