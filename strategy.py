#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return signals
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper/lower (20-period)
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # Volume confirmation: 4h volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Chopiness index (14-period) on 4h to detect ranging vs trending
    atr_14 = pd.Series(np.maximum.reduce([
        high[1:] - low[:-1],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ])).rolling(window=14, min_periods=14).mean().values
    # Pad first value
    atr_14 = np.concatenate([[np.nan], atr_14])
    
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chopiness = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    chop_denom = highest_high_14 - lowest_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid div by zero
    chop_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(chop_sum / chop_denom) / np.log10(14)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.3 * vol_ma_20[i]
        
        # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending (trend follow)
        # For breakout strategy, we want trending markets (chop < 38.2)
        trending_regime = chop[i] < 38.2
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: price breaks above 12h Donchian high + volume + trending regime
        if price_close > donch_high_aligned[i] and vol_confirm and trending_regime:
            enter_long = True
        
        # Short: price breaks below 12h Donchian low + volume + trending regime
        if price_close < donch_low_aligned[i] and vol_confirm and trending_regime:
            enter_short = True
        
        # Exit conditions: price returns to middle of Donchian channel
        donch_mid = (donch_high_aligned[i] + donch_low_aligned[i]) / 2
        exit_long = price_close < donch_mid
        exit_short = price_close > donch_mid
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
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

# Hypothesis: 12h Donchian breakout with volume confirmation and chop regime filter on 4h timeframe.
# Uses 12h Donchian channels (20-period) for major trend breaks, volume confirmation for participation,
# and chopiness filter to avoid false breakouts in ranging markets. Works in both bull and bear markets
# by catching strong directional moves. Position size 0.25 limits drawdown. Target: 80-150 trades.