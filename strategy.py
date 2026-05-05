#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
# Long when price breaks above 1d Donchian upper band AND volume > 1.3 * avg_volume(20) on 12h.
# Short when price breaks below 1d Donchian lower band AND volume > 1.3 * avg_volume(20) on 12h.
# Exit when price crosses the 1d Donchian midpoint OR ATR trailing stop is hit (signal → 0).
# Uses discrete sizing 0.25 to limit fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Donchian bands from 1d provide robust structure; volume confirmation validates breakout strength.
# ATR trailing stop adapts to volatility and reduces drawdown in bear markets like 2025.

name = "12h_1dDonchian20_VolumeConfirm_ATRStop"
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
    
    # Get 1d data ONCE before loop for Donchian bands and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed daily bars for Donchian(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) bands
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid_20 = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Calculate 1d ATR(14) for trailing stop
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # first bar: no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_20)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: volume > 1.3 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper band AND volume confirmation
            if close[i] > donchian_high_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry = close[i]
            # Short: price breaks below 1d Donchian lower band AND volume confirmation
            elif close[i] < donchian_low_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = close[i]
        elif position == 1:
            # Update highest close since entry
            if close[i] > highest_since_entry:
                highest_since_entry = close[i]
            
            # Exit long: price crosses below 1d Donchian midpoint OR ATR trailing stop hit
            if close[i] < donchian_mid_aligned[i] or close[i] < highest_since_entry - 2.5 * atr_aligned[i]:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest close since entry
            if close[i] < lowest_since_entry:
                lowest_since_entry = close[i]
            
            # Exit short: price crosses above 1d Donchian midpoint OR ATR trailing stop hit
            if close[i] > donchian_mid_aligned[i] or close[i] > lowest_since_entry + 2.5 * atr_aligned[i]:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals