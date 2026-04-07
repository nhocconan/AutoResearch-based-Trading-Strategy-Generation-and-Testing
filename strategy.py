#!/usr/bin/env python3
"""
1d_donchian_20_1w_trend_volume_v1
Hypothesis: On daily timeframe, use 20-day Donchian channels with weekly trend filter and volume confirmation.
Long when price closes above 20-day high with weekly EMA(50) trending up and volume > 1.5x 20-day average.
Short when price closes below 20-day low with weekly EMA(50) trending down and volume > 1.5x 20-day average.
Exit when price closes back inside the Donchian channel.
Designed for 10-25 trades/year to minimize fee dust while capturing strong trends with institutional validation.
Works in both bull/bear markets as Donchian adapts to volatility and weekly trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_20_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Determine weekly trend direction (using EMA slope)
    weekly_trend_up = np.zeros(len(ema_50_1w_aligned), dtype=bool)
    weekly_trend_down = np.zeros(len(ema_50_1w_aligned), dtype=bool)
    for i in range(1, len(ema_50_1w_aligned)):
        if not np.isnan(ema_50_1w_aligned[i]) and not np.isnan(ema_50_1w_aligned[i-1]):
            weekly_trend_up[i] = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_trend_down[i] = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
    
    # Calculate Donchian Channels on 1d timeframe (20-period)
    donchian_period = 20
    # Highest high over last donchian_period periods
    high_roll = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    # Lowest low over last donchian_period periods
    low_roll = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume filter: 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes back inside Donchian channel (below upper band)
            if close[i] < high_roll[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside Donchian channel (above lower band)
            if close[i] > low_roll[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and weekly trend alignment
            if vol_ok:
                # Long: price closes above 20-day high with weekly uptrend
                if (close[i] > high_roll[i] and close[i-1] <= high_roll[i-1] and 
                    weekly_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price closes below 20-day low with weekly downtrend
                elif (close[i] < low_roll[i] and close[i-1] >= low_roll[i-1] and 
                      weekly_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals