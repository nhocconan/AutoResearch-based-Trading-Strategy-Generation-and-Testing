#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND 12h EMA50 is rising AND volume > 1.5x average.
Short when price breaks below Donchian lower band AND 12h EMA50 is falling AND volume > 1.5x average.
Exit when price crosses the 12h EMA20 (mean reversion to trend).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-40 trades/year per symbol.
Donchian channels provide clear breakout levels, 12h trend filter avoids counter-trend trades in bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate EMA20 and EMA50 on 12h data
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA slope (rising/falling) on 12h data
    ema20_slope = np.diff(ema20_12h, prepend=ema20_12h[0])
    ema50_slope = np.diff(ema50_12h, prepend=ema50_12h[0])
    
    # Align 12h indicators to 4h timeframe
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema20_slope_aligned = align_htf_to_ltf(prices, df_12h, ema20_slope)
    ema50_slope_aligned = align_htf_to_ltf(prices, df_12h, ema50_slope)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(ema20_slope_aligned[i]) or np.isnan(ema50_slope_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h EMA20 slope and EMA50 slope
        trend_up = ema20_slope_aligned[i] > 0 and ema50_slope_aligned[i] > 0
        trend_down = ema20_slope_aligned[i] < 0 and ema50_slope_aligned[i] < 0
        
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper AND 12h uptrend AND volume confirmation
            if (close[i] > donchian_upper[i] and trend_up and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower AND 12h downtrend AND volume confirmation
            elif (close[i] < donchian_lower[i] and trend_down and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below 12h EMA20 (mean reversion to trend)
                if close[i] < ema20_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above 12h EMA20 (mean reversion to trend)
                if close[i] > ema20_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hEMA20_50_Trend_Volume"
timeframe = "4h"
leverage = 1.0