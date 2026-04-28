# #!/usr/bin/env python3
# Hypothesis: 1d Donchian breakout with weekly volatility filter and volume confirmation.
# Uses Donchian channels on daily timeframe with weekly ATR-based volatility filter to avoid false breakouts
# during low volatility periods. Volume confirmation ensures breakouts have participation.
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year).
# Works in both bull and bear markets by filtering for sufficient volatility and volume.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian calculation (already aligned to 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels on daily data
    high_20d = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-week ATR for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR calculation
    atr_10w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20d)
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr_10w)
    
    # Volume filter: volume > 1.3x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    # Volatility filter: current ATR > 0.7x average ATR (avoid low volatility periods)
    atr_ma = pd.Series(atr_aligned).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr_aligned > (atr_ma * 0.7)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 30)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Entry conditions with filters
        long_entry = breakout_up and volume_filter[i] and volatility_filter[i]
        short_entry = breakout_down and volume_filter[i] and volatility_filter[i]
        
        # Exit conditions: opposite breakout or volatility collapse
        long_exit = breakout_down or (not volatility_filter[i])
        short_exit = breakout_up or (not volatility_filter[i])
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_DonchianBreakout_1wATR_VolumeFilter"
timeframe = "1d"
leverage = 1.0