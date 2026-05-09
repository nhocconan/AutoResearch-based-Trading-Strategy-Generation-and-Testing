#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d price channel breakout with weekly trend filter and volume confirmation
# Uses Donchian channels on 1d for breakout signals, 1w EMA for trend filter, and volume spike for confirmation
# Designed to work in both bull and bear markets by filtering breakouts with higher timeframe trend
# Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag
name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on weekly close for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_20_1w = close_1w.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 20-period Donchian channels on daily data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (20-period average)
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma20[i]) or np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]  # Require strong volume spike
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and weekly uptrend
            if high[i] > donchian_high[i] and vol_ok and close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume and weekly downtrend
            elif low[i] < donchian_low[i] and vol_ok and close[i] < ema_20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below Donchian low or weekly trend turns down
            if low[i] < donchian_low[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above Donchian high or weekly trend turns up
            if high[i] > donchian_high[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals