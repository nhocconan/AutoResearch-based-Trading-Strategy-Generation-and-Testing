#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation
# Breakouts above 20-day high or below 20-day low capture momentum in both bull and bear markets.
# Trend filter: 1w EMA50 ensures we trade in the direction of the higher timeframe trend.
# Volume confirmation: 1d volume > 1.5x 20-day average to avoid false breakouts.
# Exit: Price returns to the 20-day moving average or trend reverses.
# Designed for 1d timeframe targeting 15-25 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channels (20-period) on 1d data
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    # Middle: 20-period SMA
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volume 20-period average for spike detection
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(sma_20[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-day high + 1w uptrend + volume spike
            if (close[i] > high_max_20[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low + 1w downtrend + volume spike
            elif (close[i] < low_min_20[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to 20-day SMA or trend reversal
            if position == 1:
                # Exit on return to SMA20 or trend reversal
                if (close[i] <= sma_20[i] or 
                    close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on return to SMA20 or trend reversal
                if (close[i] >= sma_20[i] or 
                    close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0