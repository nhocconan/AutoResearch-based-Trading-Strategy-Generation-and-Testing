#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND 1d chop > 61.8 (range regime).
# Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND 1d chop > 61.8 (range regime).
# Uses discrete position size 0.25. Donchian breakout captures momentum, volume spike confirms participation,
# chop filter ensures we trade in ranging markets where mean reversion works well.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 12h Indicators: Donchian(20) channels ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume MA calculation
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # === 1d Indicators: Choppiness Index (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    denominator = hh_1d - ll_1d
    # Avoid division by zero
    chop_1d = np.where(denominator > 0, 100 * np.log10(sum_atr_14 / denominator) / np.log10(14), 50)
    chop_1d = np.where(np.isnan(chop_1d), 50, chop_1d)
    
    # Chop > 61.8 indicates ranging regime (good for mean reversion/breakouts in range)
    chop_regime_1d = chop_1d > 61.8
    chop_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian, 20 for volume MA, 14 for chop)
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_regime_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_spike = volume_spike_1d_aligned[i]
        in_chop_regime = chop_regime_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Donchian(20) low or regime changes
            if price < lower_channel or not in_chop_regime:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Donchian(20) high or regime changes
            if price > upper_channel or not in_chop_regime:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian(20) high AND volume spike AND chop regime
            if price > upper_channel and vol_spike and in_chop_regime:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian(20) low AND volume spike AND chop regime
            elif price < lower_channel and vol_spike and in_chop_regime:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ChopRegime_V1"
timeframe = "12h"
leverage = 1.0