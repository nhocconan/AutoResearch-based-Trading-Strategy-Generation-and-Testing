#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: Daily Donchian channel breakout with weekly EMA50 trend filter and volume confirmation
    # Works in both bull and bear markets: breakouts from price channels capture directional moves
    # Weekly EMA50 filters trend direction to avoid counter-trend trades
    # Volume surge confirms breakout strength, reducing false signals
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 trend filter
    ema_1w_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Daily price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily Donchian channels (20-period)
    # Upper band: highest high of last 20 days
    # Lower band: lowest low of last 20 days
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (20-period average surge)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper band with volume surge AND weekly EMA50 uptrend
            if close[i] > highest_high[i] and vol_surge[i] and close[i] > ema_1w_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band with volume surge AND weekly EMA50 downtrend
            elif close[i] < lowest_low[i] and vol_surge[i] and close[i] < ema_1w_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Donchian band
            if position == 1:
                if close[i] < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_WeeklyEMA50_Trend_VolumeSurge_v1"
timeframe = "1d"
leverage = 1.0