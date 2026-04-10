#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above 4h Donchian upper(20) AND 1d volume > 1.3x 20-period volume SMA AND chop(14) > 61.8 (range)
# - Short when price breaks below 4h Donchian lower(20) AND 1d volume > 1.3x 20-period volume SMA AND chop(14) > 61.8 (range)
# - Exit: price retreats to Donchian midpoint or volume drops below average
# - Uses 1d timeframe for volume confirmation and chop filter, 4h for execution timing
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 15-40 trades/year on 4h timeframe to stay within fee drag limits

name = "4h_1d_donchian_volume_chop_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2.0
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1d chopiness index (14-period) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).rolling(2).max().values - pd.Series(low_1d).rolling(2).min().values
    tr2 = np.abs(pd.Series(high_1d).rolling(2).shift(1).values - pd.Series(close_1d).rolling(2).shift(1).values)
    tr3 = np.abs(pd.Series(low_1d).rolling(2).shift(1).values - pd.Series(close_1d).rolling(2).shift(1).values)
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # Chop = 100 * log10(sum(TR14) / (max(HH14) - min(LL14))) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denominator = hh_14 - ll_14
    denominator = np.where(denominator == 0, 1e-10, denominator)  # Avoid division by zero
    chop = 100 * np.log10(tr_sum / denominator) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.3x 20-period volume SMA
        idx_1d = i // 6  # 4h bars per 1d (6*4h = 24h)
        if idx_1d >= len(volume_1d):
            vol_confirm = False
        else:
            vol_confirm = volume_1d[idx_1d] > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Chop regime filter: chop > 61.8 (range-bound market)
        chop_filter = chop_aligned[i] > 61.8
        
        # Donchian breakout signals
        breakout_up = close[i] > high_max_20[i-1]  # Break above previous upper band
        breakout_down = close[i] < low_min_20[i-1]  # Break below previous lower band
        
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