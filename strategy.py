#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian breakout with volume confirmation and ADX trend filter
# Uses weekly Donchian channels to capture long-term trends, volume to confirm breakout strength,
# and ADX to ensure trending markets. Works in both bull and bear by taking breakouts
# only in the direction of the weekly trend (EMA50).
# Target: 30-100 total trades over 4 years (7-25/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF) for trend and Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    volume_weekly = df_weekly['volume'].values
    
    # Calculate Donchian channels (20-period) on weekly
    donch_high_weekly = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donch_low_weekly = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA50 on weekly for trend filter
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ADX (14-period) on weekly for trend strength
    # True Range
    tr1 = high_weekly - low_weekly
    tr2 = np.abs(high_weekly - np.roll(close_weekly, 1))
    tr3 = np.abs(low_weekly - np.roll(close_weekly, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_weekly - np.roll(high_weekly, 1)) > (np.roll(low_weekly, 1) - low_weekly),
                       np.maximum(high_weekly - np.roll(high_weekly, 1), 0), 0)
    dm_minus = np.where((np.roll(low_weekly, 1) - low_weekly) > (high_weekly - np.roll(high_weekly, 1)),
                        np.maximum(np.roll(low_weekly, 1) - low_weekly, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period on weekly)
    vol_avg_weekly = pd.Series(volume_weekly).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to daily timeframe
    donch_high_weekly_aligned = align_htf_to_ltf(prices, df_weekly, donch_high_weekly)
    donch_low_weekly_aligned = align_htf_to_ltf(prices, df_weekly, donch_low_weekly)
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    vol_avg_aligned = align_htf_to_ltf(prices, df_weekly, vol_avg_weekly)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_weekly_aligned[i]) or np.isnan(donch_low_weekly_aligned[i]) or
            np.isnan(ema50_weekly_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above weekly Donchian high + volume spike + ADX > 25 (trending) + price above weekly EMA50
        if (close[i] > donch_high_weekly_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            adx_aligned[i] > 25 and
            close[i] > ema50_weekly_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below weekly Donchian low + volume spike + ADX > 25 (trending) + price below weekly EMA50
        elif (close[i] < donch_low_weekly_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              adx_aligned[i] > 25 and
              close[i] < ema50_weekly_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or ADX < 20 (losing trend strength)
        elif position == 1 and (close[i] < donch_low_weekly_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donch_high_weekly_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian_Volume_ADX_Filter"
timeframe = "1d"
leverage = 1.0