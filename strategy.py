#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Donchian upper band (20) AND 1d volume > 2.0x 20-bar avg AND chop > 61.8 (range)
# - Short when price breaks below Donchian lower band (20) AND 1d volume > 2.0x 20-bar avg AND chop > 61.8 (range)
# - Exit when price returns to Donchian midpoint (mean reversion)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Donchian provides clear structure; volume confirms breakout strength; chop filter avoids whipsaws in strong trends

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) from 4h data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian upper/lower bands (20-period)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Pre-compute 1d volume confirmation: > 2.0x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 1d Choppiness Index (14-period) for regime filter
    # Chop = 100 * log10(sum(TR) / (ATR * n)) / log10(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR (14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index
    chop_1d = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    chop_1d = np.where(atr_14 > 0, chop_1d, 50.0)  # Handle division by zero
    
    # Chop > 61.8 indicates ranging market (good for mean reversion)
    chop_range_1d = chop_1d > 61.8
    chop_range_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_range_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(chop_range_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper AND volume spike AND chop > 61.8 (range)
            if (prices['high'].iloc[i] > donchian_upper[i] and 
                vol_spike_1d_aligned[i] and 
                chop_range_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian lower AND volume spike AND chop > 61.8 (range)
            elif (prices['low'].iloc[i] < donchian_lower[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_range_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Donchian midpoint (mean reversion)
            # Exit when price returns to Donchian midpoint
            exit_signal = False
            if position == 1:  # Long position
                if prices['low'].iloc[i] <= donchian_mid[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['high'].iloc[i] >= donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals