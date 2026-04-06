#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA(50) trend + volume confirmation
# Long when price breaks above Donchian(20) high AND 1d EMA(50) up AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND 1d EMA(50) down AND volume > 1.5x average
# Exit when price crosses opposite Donchian band or volume drops
# Uses 4h timeframe with 1d trend filter to target 75-200 total trades over 4 years
# Works in bull via breakouts, in bear via short breakdowns, volume confirms institutional interest

name = "4h_donchian20_1d_ema_vol_v2"
timeframe = "4h"
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
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donch_high = donch_high.values
    donch_low = donch_low.values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema_50 = ema_50.values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donch_low[i] or volume[i] <= volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donch_high[i] or volume[i] <= volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend and volume confirmation
            # Long: price breaks above Donchian high AND 1d EMA(50) rising AND volume confirmation
            if (close[i] > donch_high[i] and 
                ema_50_aligned[i] > ema_50_aligned[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND 1d EMA(50) falling AND volume confirmation
            elif (close[i] < donch_low[i] and 
                  ema_50_aligned[i] < ema_50_aligned[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals