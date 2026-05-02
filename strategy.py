#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
# Uses 12h primary timeframe with 1d HTF for weekly pivot and trend alignment.
# Weekly pivot (from 1d data) provides strong structural support/resistance.
# Breakouts in direction of 1d trend with volume spike capture institutional moves.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 12h timeframe.
# Works in both bull and bear markets by following the 1d trend direction only.

name = "12h_Donchian20_1dWeeklyPivot_Direction_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's high, low, close)
    # Approximate weekly from daily: use 5-day lookback for prior week
    if len(df_1d) >= 5:
        week_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(5).values
        week_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(5).values
        week_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(5).values
    else:
        week_high = np.full(len(df_1d), np.nan)
        week_low = np.full(len(df_1d), np.nan)
        week_close = np.full(len(df_1d), np.nan)
    
    # Weekly pivot = (H+L+C)/3
    week_pivot = (week_high + week_low + week_close) / 3
    week_range = week_high - week_low
    # Weekly R1/S1 (standard pivot)
    week_R1 = 2 * week_pivot - week_low
    week_S1 = 2 * week_pivot - week_high
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly pivot levels and EMA to 12h timeframe (wait for completed 1d bar)
    week_R1_aligned = align_htf_to_ltf(prices, df_1d, week_R1)
    week_S1_aligned = align_htf_to_ltf(prices, df_1d, week_S1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (2.0x 20-period average) on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 60  # max(34 for EMA, 20 for Donchian +1 for shift, 20 for volume +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(week_R1_aligned[i]) or np.isnan(week_S1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + above weekly R1 + 1d uptrend + volume spike
            if (close[i] > donchian_high[i] and close[i] > week_R1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + below weekly S1 + 1d downtrend + volume spike
            elif (close[i] < donchian_low[i] and close[i] < week_S1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns below Donchian low or weekly S1 (mean reversion) or below EMA (trend reversal)
            if close[i] < donchian_low[i] or close[i] < week_S1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns above Donchian high or weekly R1 (mean reversion) or above EMA (trend reversal)
            if close[i] > donchian_high[i] or close[i] > week_R1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals