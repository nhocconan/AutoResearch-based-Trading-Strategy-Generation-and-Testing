#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Donchian(20) breakout from 1d timeframe with volume confirmation and choppiness regime filter.
# Enter long when price breaks above 1d Donchian upper channel AND volume > 1.5x 20-bar average AND 1d choppiness < 61.8 (trending regime).
# Enter short when price breaks below 1d Donchian lower channel AND volume > 1.5x 20-bar average AND 1d choppiness < 61.8.
# Exit when price returns to the 1d Donchian midpoint or opposite breakout occurs.
# Donchian provides clear structure, volume confirms breakout strength, chop filter avoids false signals in ranging markets.
# Works in bull markets (breakouts with follow-through) and bear markets (breakdowns with continuation).
# Uses discrete position sizing (0.25) to control risk. Target: 50-150 total trades over 4 years.

name = "12h_Donchian20_1dVolumeChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian and choppiness calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    period20_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper = period20_high
    donchian_lower = period20_low
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1d choppiness index (14-period)
    # CHOP = 100 * log10(sum(ATR14) / (log10(HH14 - LL14) / log10(14)))
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = hh14 - ll14
    chop = np.where(
        (chop_denom > 0) & (~np.isnan(atr14)) & (~np.isnan(chop_denom)),
        100 * np.log10(np.nansum(atr14[-13:]) / (np.log10(chop_denom) / np.log10(14))),
        np.nan
    )
    # For simplicity, use rolling sum of ATR14
    atr_sum_14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum_14 / (np.log10(chop_denom) / np.log10(14)))
    chop = np.where(chop_denom > 0, chop, np.nan)
    
    # Align 1d indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_upper_aligned[i]
        bearish_breakout = close[i] < donchian_lower_aligned[i]
        
        # Entry conditions
        long_entry = bullish_breakout and volume_confirm[i] and trending_regime
        short_entry = bearish_breakout and volume_confirm[i] and trending_regime
        
        # Exit conditions: return to midpoint or opposite breakout
        long_exit = close[i] < donchian_mid_aligned[i] or bearish_breakout
        short_exit = close[i] > donchian_mid_aligned[i] or bullish_breakout
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals