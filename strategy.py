#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Donchian upper band (20-period high) AND 1d volume > 1.5x 20-period volume SMA AND 1d chop > 61.8 (range regime)
# - Short when price breaks below Donchian lower band (20-period low) AND 1d volume > 1.5x 20-period volume SMA AND 1d chop > 61.8 (range regime)
# - Exit: price retreats to Donchian midpoint or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 15-40 trades/year on 4h timeframe to stay within fee drag limits
# - Uses Donchian channels for structure, volume spike for confirmation, chop filter to avoid trending markets

name = "4h_1d_donchian_volume_chop_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1d OHLC for chop and volume
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d True Range for chop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR
    
    # Calculate 1d ATR (14-period) for chop denominator
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Chopiness Index (14-period)
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop_denominator = atr_14 * 14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # Avoid division by zero
    chop_ratio = sum_tr_14 / chop_denominator
    chop_ratio = np.where(chop_ratio <= 0, 1e-10, chop_ratio)  # Avoid log of non-positive
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Align 1d indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1d volume SMA for confirmation
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 4h volume for confirmation
    volume_sma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(volume_sma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.5x 20-period volume SMA AND 1d volume > 1.5x 20-period volume SMA
        vol_confirm_4h = volume[i] > 1.5 * volume_sma_20_4h[i]
        vol_confirm_1d = volume_1d[i // 6] > 1.5 * volume_sma_20_1d_aligned[i] if i // 6 < len(volume_1d) else False
        vol_confirm = vol_confirm_4h and vol_confirm_1d
        
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion/breakout fade)
        chop_filter = chop_aligned[i] > 61.8
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper_aligned[i-1]  # Break above previous upper band
        breakout_down = close[i] < donchian_lower_aligned[i-1]  # Break below previous lower band
        
        # Exit conditions: price retreats to midpoint or loss of confirmation
        exit_long = close[i] < donchian_mid_aligned[i] or not vol_confirm or not chop_filter
        exit_short = close[i] > donchian_mid_aligned[i] or not vol_confirm or not chop_filter
        
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