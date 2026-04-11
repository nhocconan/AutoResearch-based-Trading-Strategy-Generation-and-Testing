#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_4h_donchian_volume_regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # 1d Donchian(20) - previous completed day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_1d = np.roll(high_20_1d, 1)
    donch_low_1d = np.roll(low_20_1d, 1)
    donch_high_1d[0] = np.nan
    donch_low_1d[0] = np.nan
    
    # Align to 4h
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # 4h Donchian(20) breakout
    high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness regime filter: CHOP(14) > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
    # Calculate CHOP on 4h
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first element has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr14 * 14) / (highest_high - lowest_low)) / np.log10(14)
    chop[np.isnan(chop)] = 50  # neutral when undefined
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high_1d_aligned[i]) or np.isnan(donch_low_1d_aligned[i]) or
            np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        trending_regime = chop[i] < 38.2
        
        # Long: price breaks above 1d Donchian high with volume and trending regime
        long_signal = volume_confirmed and trending_regime and (price_high > donch_high_1d_aligned[i])
        
        # Short: price breaks below 1d Donchian low with volume and trending regime
        short_signal = volume_confirmed and trending_regime and (price_low < donch_low_1d_aligned[i])
        
        # Exit: price crosses back inside 4h Donchian channels
        exit_long = position == 1 and price_close < low_4h[i]
        exit_short = position == -1 and price_close > high_4h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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

# Hypothesis: 4h Donchian breakout with 1d Donchian filter, volume confirmation, and chop regime filter.
# Uses 1d Donchian(20) from previous completed day as higher timeframe trend filter.
# Enters long when 4h price breaks above 1d Donchian high with volume (>1.8x average) in trending regime (CHOP<38.2).
# Enters short when 4h price breaks below 1d Donchian low with volume and trending regime.
# Exits when price returns inside 4h Donchian(20) channels.
# The 1d Donchian filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation reduces false breakouts.
# Chop regime filter avoids whipsaws in ranging markets.
# Designed for low trade frequency (target: 25-50 trades/year) to minimize fee drag on 4h timeframe.
# Works in both bull and bear markets by trading breakouts in the direction of the higher timeframe trend.