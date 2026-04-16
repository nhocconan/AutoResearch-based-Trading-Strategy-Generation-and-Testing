#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h ATR expansion breakout with volume confirmation and ATR trailing stop.
# ATR expansion indicates increasing volatility and potential trend continuation.
# Long when price breaks above 12h Donchian upper + ATR(12h) > 1.2x its 20-period median + volume > 1.5x median.
# Short when price breaks below 12h Donchian lower + ATR(12h) > 1.2x its 20-period median + volume > 1.5x median.
# Uses discrete position size 0.25. ATR trailing stop exits when price moves against position by 2.5x ATR.
# ATR expansion filter ensures we only trade during volatile breakout periods, reducing false signals in ranging markets.
# Works in both bull and bear markets by capturing strong directional moves with volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Donchian channels, ATR, and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 12h Indicators: Donchian Channels (20-period) ===
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: ATR (14-period) for expansion filter ===
    tr1_12h = high_12h[1:] - low_12h[1:]
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_median_20 = pd.Series(atr_12h).rolling(window=20, min_periods=20).median().values
    
    # === 12h Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).median().values
    
    # === 4h Indicators: ATR (14-period) for trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Align all indicators to primary timeframe (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    atr_median_aligned = align_htf_to_ltf(prices, df_12h, atr_median_20)
    vol_median_aligned = align_htf_to_ltf(prices, df_12h, vol_median_20)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 20  # Donchian, ATR median, and volume median need 20 periods
    
    # Track position state and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(atr_12h_aligned[i]) or np.isnan(atr_median_aligned[i]) or
            np.isnan(vol_median_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        atr_12h_val = atr_12h_aligned[i]
        atr_median_val = atr_median_aligned[i]
        vol_median_val = vol_median_aligned[i]
        price = close[i]
        atr_val = atr[i]
        current_vol_12h = volume_12h_aligned[i]
        
        # ATR expansion filter: current 12h ATR > 1.2x its 20-period median
        atr_expansion = atr_12h_val > (atr_median_val * 1.2)
        
        # Volume spike filter: current 12h volume > 1.5x median volume
        volume_spike = current_vol_12h > (vol_median_val * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # ATR trailing stop: exit if price drops below entry_price - 2.5 * ATR
            if price < entry_price - 2.5 * atr_val:
                exit_signal = True
            # Optional: exit on reverse breakout with volume and ATR expansion
            elif price < lower and volume_spike and atr_expansion:
                exit_signal = True
        
        elif position == -1:  # Short position
            # ATR trailing stop: exit if price rises above entry_price + 2.5 * ATR
            if price > entry_price + 2.5 * atr_val:
                exit_signal = True
            # Optional: exit on reverse breakout with volume and ATR expansion
            elif price > upper and volume_spike and atr_expansion:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Donchian upper with volume spike AND ATR expansion
            if price > upper and volume_spike and atr_expansion:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian lower with volume spike AND ATR expansion
            elif price < lower and volume_spike and atr_expansion:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_12hATRExpansion_Donchian20_VolumeSpike1.5x_ATRTrail2.5_v1"
timeframe = "4h"
leverage = 1.0