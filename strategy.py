#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian breakout with volume confirmation and trend filter
# Uses 1-month EMA for trend direction to avoid counter-trend trades
# Breakout at weekly Donchian channels provides strong signal with fewer false signals
# Designed for 1d timeframe to target 20-50 trades/year per symbol, works in bull/bear via trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channels and EMA
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels on weekly high/low
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period EMA on weekly close for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_20 = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike filter (20-period on daily)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian + volume spike + uptrend (close > EMA20)
            if (close[i] > high_20_aligned[i] and vol_spike[i] and close[i] > ema_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian + volume spike + downtrend (close < EMA20)
            elif (close[i] < low_20_aligned[i] and vol_spike[i] and close[i] < ema_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if close[i] < low_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyTrend_Volume_Session"
timeframe = "1d"
leverage = 1.0