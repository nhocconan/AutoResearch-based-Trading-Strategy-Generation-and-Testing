#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trading with 4h/1d trend filter and volume confirmation
# Uses 4h Donchian breakout for direction, 1h for entry timing with volume spike
# Session filter (08-20 UTC) to reduce noise. Target: 15-37 trades/year per symbol
# Works in bull/bear via 4h trend filter - only trade in direction of 4h trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4-hour data for trend and Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1-day data for additional trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_ma20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_ma20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h EMA20 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema20_4h = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for stronger trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike detection (20-period on 1h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align 4h indicators to 1h timeframe
    high_ma20_aligned = align_htf_to_ltf(prices, df_4h, high_ma20)
    low_ma20_aligned = align_htf_to_ltf(prices, df_4h, low_ma20)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Align 1d indicators to 1h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(high_ma20_aligned[i]) or np.isnan(low_ma20_aligned[i]) or
            np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high + volume spike + 4h/1d uptrend
            if (close[i] > high_ma20_aligned[i] and vol_spike[i] and 
                close[i] > ema20_4h_aligned[i] and close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h Donchian low + volume spike + 4h/1d downtrend
            elif (close[i] < low_ma20_aligned[i] and vol_spike[i] and 
                  close[i] < ema20_4h_aligned[i] and close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level or trend weakens
            if position == 1:
                if (close[i] < low_ma20_aligned[i] or 
                    close[i] < ema20_4h_aligned[i] or close[i] < ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if (close[i] > high_ma20_aligned[i] or 
                    close[i] > ema20_4h_aligned[i] or close[i] > ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4h1dTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0