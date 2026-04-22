#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# Donchian channels provide robust breakout levels in trending and ranging markets.
# 1d EMA50 ensures alignment with daily trend for higher probability trades.
# Volume confirmation (>1.5x 30-period average) filters false breakouts.
# Designed for 12h timeframe targeting 15-30 trades/year with strong performance in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + daily uptrend + volume confirmation
            if (close[i] > high_max[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_30[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + daily downtrend + volume confirmation
            elif (close[i] < low_min[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_30[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or trend reversal
            if position == 1:
                # Exit long: price returns below Donchian low or trend turns down
                if (close[i] < low_min[i] or 
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns above Donchian high or trend turns up
                if (close[i] > high_max[i] or 
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0