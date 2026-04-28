#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with 1d EMA trend filter and volume confirmation.
# Uses 1h timeframe for entry timing precision, 4h Donchian(20) for structure/breakout levels,
# and 1d EMA50 for trend filter (works in both bull/bear by capturing intermediate trend).
# Volume spike (>2.0x 20-bar average) confirms breakout strength.
# Session filter (08-20 UTC) reduces noise trades outside active hours.
# Position size 0.20 balances return and drawdown. Discrete levels minimize fee churn.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h (within proven winning range).

name = "1h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (structure/breakout levels)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper = max(high, lookback=20)
    high_roll_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower = min(low, lookback=20)
    low_roll_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (use previous 4h bar's levels)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, high_roll_max)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, low_roll_min)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 1h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h volume spike: >2.0x 20-bar average volume (stricter to reduce trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or
            not (8 <= hours[i] <= 20)):  # Session filter: 08-20 UTC
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Donchian breakout conditions with volume confirmation
        long_breakout = close[i] > donchian_upper_aligned[i] and volume_spike[i]
        short_breakout = close[i] < donchian_lower_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Donchian level or trend reversal
        long_exit = close[i] < donchian_lower_aligned[i] or below_ema
        short_exit = close[i] > donchian_upper_aligned[i] or above_ema
        
        # Handle entries and exits
        if long_breakout and above_ema and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_breakout and below_ema and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals