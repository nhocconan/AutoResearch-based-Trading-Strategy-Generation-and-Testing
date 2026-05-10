#!/usr/bin/env python3
# 1D_WeeklyDonchian_Breakout_Volume
# Hypothesis: Weekly Donchian channel breakouts capture long-term trends. Volume confirmation ensures participation, and a 1-day EMA filter reduces whipsaws. This strategy targets 20-30 trades/year on daily timeframe to minimize fee drag while capturing major moves in both bull and bear markets.

name = "1D_WeeklyDonchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # 1-day EMA filter (50-period) for trend direction
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA and volume MA to stabilize
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(ema_50[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high, above EMA50, with volume
            if close[i] > donchian_high_aligned[i] and close[i] > ema_50[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low, below EMA50, with volume
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_50[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals