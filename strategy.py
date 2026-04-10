#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 1d chop regime filter
# - Long when price breaks above Donchian upper band (20-period high) AND 12h volume > 1.25x 20-period volume SMA AND 1d chop > 61.8 (ranging market)
# - Short when price breaks below Donchian lower band (20-period low) AND 12h volume > 1.25x 20-period volume SMA AND 1d chop > 61.8
# - Exit: price crosses Donchian midpoint (mean of upper and lower band)
# - Uses 4h for price action (Donchian channels), 12h for volume confirmation, 1d for chop filter
# - Target: 19-50 trades/year to minimize fee drag while capturing high-probability breakouts in ranging markets
# - Volume confirmation on 12h reduces noise vs 1d; chop filter on 1d ensures ranging regime for mean-reversion exits

name = "4h_12h_1d_donchian_volspike_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for volume confirmation (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Load 1d data ONCE before loop for chop filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 12h volume SMA for confirmation
    vol_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Pre-compute 1d Chopiness Index (14-period) for regime filter
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    tr1 = np.abs(df_1d_high[1:] - df_1d_low[:-1])
    tr2 = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr3 = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(df_1d_high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d_low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop = np.where((highest_high_14 - lowest_low_14) == 0, 50, chop)  # avoid division by zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_sma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
        
        # Volume confirmation: 12h volume > 1.25x 20-period volume SMA
        vol_confirm = vol_12h_aligned[i] > 1.25 * volume_sma_20_12h_aligned[i]
        
        # Chop filter: chop > 61.8 indicates ranging market (good for breakout mean reversion)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        # Only trade when both volume confirmation and chop filter are present
        if vol_confirm and chop_filter:
            # Long: price breaks above Donchian upper band
            if close[i] > donchian_upper[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: price breaks below Donchian lower band
            elif close[i] < donchian_lower[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            
            # Exit conditions: price crosses Donchian midpoint
            if position == 1 and close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals