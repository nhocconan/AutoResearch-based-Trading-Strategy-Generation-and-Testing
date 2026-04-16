#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout with volume confirmation and ATR-based stoploss.
# Long when price breaks above 1d Donchian upper channel (20-period) + volume > 1.5x 20-period median volume.
# Short when price breaks below 1d Donchian lower channel (20-period) + volume > 1.5x 20-period median volume.
# Exit on opposite Donchian channel touch or when ATR-based trailing stop is hit.
# Uses discrete position size 0.25. Donchian channels capture volatility and breakouts in trending markets.
# Volume spike ensures institutional participation. 4h timeframe targets 20-50 trades/year to minimize fee drag.
# Works in bull markets (breakouts) and bear markets (breakdowns) with volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    vol_1d = df_1d['volume'].values
    
    # === 1d Indicators: Donchian Channels (20-period) ===
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # === 1d Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    
    # === 1d Indicators: ATR (14-period) for stoploss ===
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first period has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 20  # Donchian and volume median need 20 periods
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values (aligned)
        price = close[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_aligned[i]
        
        # Get current 1d volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 1.5x median volume
        volume_spike = current_vol_1d > (vol_median * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price touches lower Donchian channel OR ATR stoploss hit
            if (price <= lower) or (price <= entry_price - 2.0 * atr):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price touches upper Donchian channel OR ATR stoploss hit
            if (price >= upper) or (price >= entry_price + 2.0 * atr):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper Donchian channel + volume spike
            if (price > upper) and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below lower Donchian channel + volume spike
            elif (price < lower) and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_1dDonchian20_VolumeSpike1.5x_ATRStop2.0_V1"
timeframe = "4h"
leverage = 1.0