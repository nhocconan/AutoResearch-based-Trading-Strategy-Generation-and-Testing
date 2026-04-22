#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter + weekly Donchian breakout + volume confirmation.
# Uses weekly Donchian channel (20-period) for directional bias, daily Choppiness Index (14-period) to filter ranging markets.
# Long when price breaks above weekly Donchian high AND chop > 61.8 (ranging) for mean reversion setup.
# Short when price breaks below weekly Donchian low AND chop > 61.8.
# Volume confirmation reduces false breakouts.
# Designed for 1d timeframe to capture multi-day swings with low frequency.
# Target: 10-25 trades/year per symbol (40-100 total) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channel (trend filter)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian channel: 20-period high/low (prior week)
    high_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Shift to use only completed weeks (avoid look-ahead)
    high_20w = np.roll(high_20w, 1)
    low_20w = np.roll(low_20w, 1)
    
    # Load daily data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range (TR)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no prior close)
    tr[0] = high_1d[0] - low_1d[0]
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(sumTR / (ATR * 14)) / log10(14)
    # Avoid division by zero or invalid values
    atr14 = atr * 14
    # Only compute where valid
    chop = np.full_like(close, np.nan, dtype=float)
    mask = (sum_tr > 0) & (atr14 > 0)
    chop[mask] = 100 * np.log10(sum_tr[mask] / atr14[mask]) / np.log10(14)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # Align indicators to daily timeframe
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + chop > 61.8 (ranging) + volume spike
            if (close[i] > high_20w_aligned[i] and 
                chop_aligned[i] > 61.8 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low + chop > 61.8 (ranging) + volume spike
            elif (close[i] < low_20w_aligned[i] and 
                  chop_aligned[i] > 61.8 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price breaks opposite weekly Donchian level or chop < 38.2 (trending)
            if position == 1:
                if (close[i] < low_20w_aligned[i] or chop_aligned[i] < 38.2):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > high_20w_aligned[i] or chop_aligned[i] < 38.2):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Choppiness_WeeklyDonchian_Breakout_Volume_Spike"
timeframe = "1d"
leverage = 1.0