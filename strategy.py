#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily volatility-adjusted Donchian breakouts with volume confirmation
# and ADX trend filter. Long when price breaks above upper Donchian(20) with volume > 1.5x average
# and ADX > 25. Short when price breaks below lower Donchian(20) with volume > 1.5x average and ADX > 25.
# Exit when price returns to the middle of the Donchian channel or ADX drops below 20.
# Volatility adjustment: Donchian bands scaled by ATR(14) to adapt to changing market conditions.
# This adapts to both high and low volatility regimes, reducing false breakouts.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for ATR and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for ATR(14) and ADX(14)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR (14)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR = smoothed TR (Wilder's smoothing)
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr[13] = np.nanmean(tr[1:14])  # First ATR: simple average of first 14 TR
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate ADX (14)
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Base Donchian width
    donchian_width = highest_high - lowest_low
    middle = (highest_high + lowest_low) / 2
    
    # Volatility-adjusted Donchian bands
    # Scale bands by ATR to adapt to volatility
    atr_scaled = atr_aligned * 0.5  # Scale factor for band width
    upper_band = middle + (donchian_width / 2) + atr_scaled
    lower_band = middle - (donchian_width / 2) - atr_scaled
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(lookback, 34)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or
            np.isnan(atr_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for volatility-adjusted Donchian breakouts with volume confirmation
            # Long: price breaks above upper band AND volume confirmation AND strong trend
            if (close[i] > upper_band[i] and 
                volume_confirm[i] and
                adx_aligned[i] > 25):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower band AND volume confirmation AND strong trend
            elif (close[i] < lower_band[i] and 
                  volume_confirm[i] and
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle of channel or trend weakens
            if (close[i] <= middle[i] or 
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle of channel or trend weakens
            if (close[i] >= middle[i] or 
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_volatility_adjusted_donchian_breakout_volume_adx"
timeframe = "4h"
leverage = 1.0