#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND choppiness index < 38.2 (trending regime).
# Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND choppiness index < 38.2 (trending regime).
# Uses discrete position size 0.25. Donchian breakouts capture momentum, volume confirmation ensures participation,
# choppiness filter avoids ranging markets. Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    
    # === 4h Indicators: Choppiness Index (14-period) ===
    # Chop = 100 * log10(sum(ATR(1) over n) / (log10(n) * (highest_high - lowest_low)))
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first bar TR
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values  # ATR(1) = TR
    sum_atr = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (np.log10(14) * (highest_high_14 - lowest_low_14)))
    chop = np.where((highest_high_14 - lowest_low_14) == 0, 50, chop)  # avoid division by zero
    
    # Get 1d data once before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume MA calculation
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # Align 1d volume spike to 4h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian, 14 for Chop, 20 for volume MA)
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(chop[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        dch_high = donchian_high[i]
        dch_low = donchian_low[i]
        chop_val = chop[i]
        vol_spike = volume_spike_1d_aligned[i] > 0.5  # convert back to boolean
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low or choppiness exceeds 61.8 (ranging regime)
            if price < dch_low or chop_val > 61.8:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high or choppiness exceeds 61.8 (ranging regime)
            if price > dch_high or chop_val > 61.8:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike AND chop < 38.2 (trending)
            if price > dch_high and vol_spike and chop_val < 38.2:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian low AND volume spike AND chop < 38.2 (trending)
            elif price < dch_low and vol_spike and chop_val < 38.2:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0