#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above 4h Donchian upper(20) AND 1d volume > 1.5x 20-period volume SMA AND chop(14) < 38.2 (trending)
# - Short when price breaks below 4h Donchian lower(20) AND 1d volume > 1.5x 20-period volume SMA AND chop(14) < 38.2 (trending)
# - Exit: price retreats to 4h Donchian midpoint or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 20-50 trades/year on 4h timeframe to stay within fee drag limits
# - Uses Donchian channels for structure, volume for confirmation, chop regime to avoid ranging markets

name = "4h_1d_donchian_volume_chop_v5"
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
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian channels (20-period)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll_max
    donchian_lower = low_roll_min
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 4h chop regime filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_high_low = pd.Series(high + low).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_high_low / (atr * 14)) / np.log10(14)
    chop_aligned = chop  # Already LTF, no alignment needed
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        # Map 4h index to 1d index (approx: 6 4h bars per 1d)
        idx_1d = i // 6
        vol_confirm = (idx_1d < len(volume_1d) and 
                      volume_1d[idx_1d] > 1.5 * volume_sma_20_1d_aligned[i])
        
        # Chop regime filter: trending market (chop < 38.2)
        chop_filter = chop_aligned[i] < 38.2
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_down = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # Exit conditions: price retreats to midpoint or loss of volume confirmation
        exit_long = close[i] < donchian_mid[i] or not vol_confirm
        exit_short = close[i] > donchian_mid[i] or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm and chop_filter:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm and chop_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals