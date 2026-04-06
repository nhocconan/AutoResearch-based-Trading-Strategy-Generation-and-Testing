#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1w EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND price < 1w EMA(50) AND volume > 1.5x average
# Exit when price crosses the 1w EMA(50) or after 5 days
# Designed for trending markets with clear breaks, works in both bull/bear by following trend
# Target: 30-100 trades over 4 years (7-25/year)

name = "1d_donchian_1w_ema_vol_v2"
timeframe = "1d"
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
    
    # Donchian Channel (20-period) on daily data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # Weekly EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    
    ema_50 = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema_50 = ema_50.values
    
    # Align weekly EMA to daily timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Exit conditions: price crosses EMA(50) OR max 5 days held
        if position != 0:
            bars_since_entry += 1
            if bars_since_entry >= 5:  # time-based exit
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            elif position == 1 and close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            elif position == -1 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = position * 0.25
        else:
            # Look for entries: Donchian breakout with trend filter and volume
            # Long: break above Donchian high AND price above weekly EMA(50) + volume
            if close[i] > donchian_high[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: break below Donchian low AND price below weekly EMA(50) + volume
            elif close[i] < donchian_low[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
    
    return signals