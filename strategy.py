#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when: Price breaks above 20-day high AND 1w EMA50 is rising AND volume > 1.5x 20-day avg volume
# Short when: Price breaks below 20-day low AND 1w EMA50 is falling AND volume > 1.5x 20-day avg volume
# Exit when price returns to 20-day midpoint (mean reversion)
# Donchian breakout captures volatility expansion after consolidation
# 1w EMA50 filter ensures we trade in direction of higher timeframe trend
# Volume confirmation reduces false breakouts
# Works in both bull and bear markets by trading breakouts in direction of weekly trend
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25

name = "1d_Donchian20_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA(50)
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w EMA50 slope (rising/falling)
    ema_50_slope = np.diff(ema_50_1w_aligned, prepend=ema_50_1w_aligned[0])
    ema_50_rising = ema_50_slope > 0
    ema_50_falling = ema_50_slope < 0
    
    # Calculate Donchian channels (20) on 1d
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2
    
    # Calculate 20-day average volume for confirmation
    avg_vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_vol_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(avg_vol_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 20-day high AND weekly EMA50 rising AND volume spike
            if close[i] > highest_20[i] and ema_50_rising[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below 20-day low AND weekly EMA50 falling AND volume spike
            elif close[i] < lowest_20[i] and ema_50_falling[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to Donchian midpoint (mean reversion)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to Donchian midpoint (mean reversion)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals