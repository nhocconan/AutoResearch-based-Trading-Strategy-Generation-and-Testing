#!/usr/bin/env python3
# 1d_1w_WeeklyDonchianBreakout_TrendFollow
# Hypothesis: Use 1-week Donchian breakouts (20-week high/low) for direction, filtered by 1-day trend (EMA50) and volume.
# Enters long when price breaks above weekly high with daily uptrend and volume confirmation.
# Enters short when price breaks below weekly low with daily downtrend and volume confirmation.
# Designed for low trade frequency (~10-30 trades/year) to minimize fee drag while capturing long-term trends.
# Works in bull/bear markets by following weekly structure with daily confirmation.

name = "1d_1w_WeeklyDonchianBreakout_TrendFollow"
timeframe = "1d"
leverage = 1.0

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
    
    # Volume spike: >1.5x 50-period average (on daily timeframe)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Weekly data for Donchian channels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-week high/low)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)  # Align EMA50 to weekly for consistency
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly high + daily EMA50 uptrend + volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly low + daily EMA50 downtrend + volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly low OR closes below EMA50
            if (close[i] < donchian_low_aligned[i]) or \
               (close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly high OR closes above EMA50
            if (close[i] > donchian_high_aligned[i]) or \
               (close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals