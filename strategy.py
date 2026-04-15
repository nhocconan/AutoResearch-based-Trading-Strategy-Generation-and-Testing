#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (EMA50), volume confirmation, and volatility-based exits.
# Uses breakouts of 4h Donchian channels for trend capture, confirmed by volume spikes and 12h EMA trend direction.
# Exits when price reverses to opposite Donchian band or volatility expands (ATR-based stop).
# Designed to work in both bull and bear markets by only taking breakouts in the direction of the 12h trend.
# Target: 100-200 total trades over 4 years (25-50/year) with disciplined, high-conviction entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for price action and Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    donch_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA50 on 12h for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR (14-period) on 4h for volatility-based exits
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Average volume (20-period) on 4h for confirmation
    vol_avg_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    vol_avg_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_4h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_4h_aligned[i]) or np.isnan(donch_low_4h_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + price above 12h EMA50
        if (close[i] > donch_high_4h_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            close[i] > ema50_12h_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + price below 12h EMA50
        elif (close[i] < donch_low_4h_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              close[i] < ema50_12h_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price reaches opposite Donchian band (mean reversion within channel)
        elif position == 1 and close[i] < donch_low_4h_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donch_high_4h_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_12hEMA_Volume_Breakout"
timeframe = "4h"
leverage = 1.0