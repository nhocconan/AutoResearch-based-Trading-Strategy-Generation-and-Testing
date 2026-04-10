#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation
# - Long when price breaks above 20-period Donchian high (4h) AND 1d ATR ratio (current/20-period MA) > 1.2 AND volume > 1.5x 20-period average
# - Short when price breaks below 20-period Donchian low (4h) AND 1d ATR ratio > 1.2 AND volume > 1.5x 20-period average
# - Exit when price crosses 20-period Donchian midpoint or opposite breakout occurs
# - ATR ratio filter ensures we trade during elevated volatility regimes (avoids low-vol chop)
# - Volume confirmation prevents false signals in low participation
# - Target: 19-50 trades/year on 4h (75-200 total over 4 years) to avoid fee drag

name = "4h_1d_donchian_breakout_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d ATR(14) and its 20-period MA for ratio
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14) - Wilder's smoothing
    atr_14 = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= 14:
        atr_14[13] = np.nanmean(tr[1:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # ATR 20-period MA
    atr_ma_20 = np.full_like(atr_14, np.nan, dtype=float)
    for i in range(19, len(atr_14)):
        atr_ma_20[i] = np.nanmean(atr_14[i-19:i+1])
    
    # ATR ratio (current ATR / 20-period MA) - >1.2 indicates elevated volatility
    atr_ratio = np.full_like(atr_14, np.nan, dtype=float)
    mask = ~np.isnan(atr_14) & ~np.isnan(atr_ma_20) & (atr_ma_20 != 0)
    atr_ratio[mask] = atr_14[mask] / atr_ma_20[mask]
    
    # Align HTF indicators to 4h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian high and low
    donchian_high = np.full_like(close_4h, np.nan, dtype=float)
    donchian_low = np.full_like(close_4h, np.nan, dtype=float)
    donchian_mid = np.full_like(close_4h, np.nan, dtype=float)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high_4h[i-19:i+1])
        donchian_low[i] = np.min(low_4h[i-19:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Pre-compute 4h volume 20-period MA
    vol_4h = prices['volume'].values
    vol_ma_20 = np.full_like(vol_4h, np.nan, dtype=float)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(vol_4h[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr_ratio_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_spike = vol_4h[i] > 1.5 * vol_ma_20[i]
        
        close_now = close_4h[i]
        donchian_high_now = donchian_high[i]
        donchian_low_now = donchian_low[i]
        donchian_mid_now = donchian_mid[i]
        atr_ratio_now = atr_ratio_1d_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close_now > donchian_high_now  # price breaks above Donchian high
        breakout_down = close_now < donchian_low_now  # price breaks below Donchian low
        mid_cross_up = (close_4h[i-1] <= donchian_mid_now and close_now > donchian_mid_now)  # crosses above midpoint
        mid_cross_down = (close_4h[i-1] >= donchian_mid_now and close_now < donchian_mid_now)  # crosses below midpoint
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND ATR ratio > 1.2 (elevated vol) AND volume spike
            if (breakout_up and atr_ratio_now > 1.2 and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND ATR ratio > 1.2 (elevated vol) AND volume spike
            elif (breakout_down and atr_ratio_now > 1.2 and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Donchian midpoint or opposite Donchian breakout
            exit_long = (position == 1 and 
                        (mid_cross_down or breakout_down))
            exit_short = (position == -1 and 
                         (mid_cross_up or breakout_up))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals