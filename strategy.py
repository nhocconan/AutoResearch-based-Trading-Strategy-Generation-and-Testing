#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-week Donchian breakout + 1-day ATR filter + volume confirmation.
# Uses weekly Donchian channels (20-period) for trend direction and daily ATR for volatility filtering.
# Long when price breaks above weekly Donchian high with volume > 1.5x daily average and ATR > daily ATR average.
# Short when price breaks below weekly Donchian low with same conditions.
# Designed for low trade frequency (~20-40/year) to minimize fee decay while capturing major trends.
# Works in bull markets by catching breakouts and in bear markets by avoiding false signals through volatility filter.

name = "4h_1w_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Load daily data for ATR and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly and daily data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure Donchian channels are valid
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # ATR filter: current ATR > daily ATR average (avoid low volatility periods)
        atr_filter = atr_1d_aligned[i] > np.nanmean(atr_1d_aligned[max(0, i-50):i+1]) if i >= 50 else True
        
        # Entry conditions: price breaks Donchian levels with volume and ATR confirmation
        long_breakout = high[i] > donchian_high_aligned[i]
        short_breakout = low[i] < donchian_low_aligned[i]
        
        long_entry = long_breakout and vol_filter and atr_filter
        short_entry = short_breakout and vol_filter and atr_filter
        
        # Exit conditions: price returns to opposite Donchian level
        exit_long = low[i] < donchian_low_aligned[i] if not np.isnan(donchian_low_aligned[i]) else False
        exit_short = high[i] > donchian_high_aligned[i] if not np.isnan(donchian_high_aligned[i]) else False
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals