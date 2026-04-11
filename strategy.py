#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + chop regime filter
# - Long when price breaks above Donchian(20) high AND volume > 1.5x 20-period volume SMA AND chop > 61.8 (range)
# - Short when price breaks below Donchian(20) low AND volume > 1.5x 20-period volume SMA AND chop > 61.8 (range)
# - Exit when price crosses Donchian midpoint OR chop < 38.2 (trend)
# - Uses 1d HTF for chop filter to avoid lower timeframe noise
# - Target: 20-40 trades/year to minimize fee drag while capturing breakouts in ranging markets

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for chop filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d Chop Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR
    
    # ATR and highest high/lowest low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop Index formula: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = highest_high_14 - lowest_low_14
    # Avoid division by zero
    chop = np.where(hh_ll_diff > 0, 100 * np.log10(tr_sum_14 / hh_ll_diff) / np.log10(14), 50)
    chop[np.isnan(chop)] = 50
    
    # Align Chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2
    
    # Pre-compute 4h volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(chop_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Chop regime filter: chop > 61.8 indicates ranging market (good for mean reversion/breakout fade)
        # But we want breakouts in ranging markets, so we use chop > 61.8 to confirm we're in a range
        chop_filter = chop_aligned[i] > 61.8
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_20[i-1]  # Price breaks above previous Donchian high
        breakout_down = close[i] < lowest_low_20[i-1]  # Price breaks below previous Donchian low
        
        # Entry conditions
        enter_long = breakout_up and vol_confirm and chop_filter
        enter_short = breakout_down and vol_confirm and chop_filter
        
        # Exit conditions
        exit_long = close[i] < donchian_mid[i] or chop_aligned[i] < 38.2  # Exit when price crosses midpoint or trend emerges
        exit_short = close[i] > donchian_mid[i] or chop_aligned[i] < 38.2  # Exit when price crosses midpoint or trend emerges
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals