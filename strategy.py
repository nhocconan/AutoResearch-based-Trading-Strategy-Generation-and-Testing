#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d trend filter (EMA50) and volume confirmation.
# Buy when price breaks above 20-period high AND price > 1d EMA50 AND volume > 1.5x 20-period average.
# Sell when price breaks below 20-period low OR price < 1d EMA50.
# Short when price breaks below 20-period low AND price < 1d EMA50 AND volume > 1.5x 20-period average.
# Cover when price breaks above 20-period high OR price > 1d EMA50.
# Uses 4h timeframe for entries/exits, 1d for trend filter.
# Designed to work in both bull and bear markets by filtering trades with higher timeframe trend.
# Target: 20-50 trades/year to minimize fee drag.
name = "4h_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high AND price above 1d EMA50 AND volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low AND price below 1d EMA50 AND volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below Donchian low OR price below 1d EMA50
            if close[i] < donchian_low[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above Donchian high OR price above 1d EMA50
            if close[i] > donchian_high[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals