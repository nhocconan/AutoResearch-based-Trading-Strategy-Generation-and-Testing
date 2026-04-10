#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# - Donchian(20) from 4h: upper/lower bands for breakout detection
# - 1d volume spike: current 4h volume > 1.5x 20-period average of 4h volume (using 1d HTF for volume average stability)
# - Choppiness regime: CHOP(14) from 1d < 38.2 = trending market (favor breakouts), > 61.8 = ranging (avoid)
# - Designed for 4h timeframe: targets 20-50 trades/year (75-200 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: chop regime filter ensures we only trade breakouts in trending conditions
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(20)

name = "4h_1d_donchian_volume_chop_v1"
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
    
    # Pre-compute 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR(14) - smoothed TR
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: CHOP = 100 * log10(sum(TR(14)) / (HH(14) - LL(14))) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_14 - ll_14
    chop = 100 * np.log10(sum_tr_14 / (range_14 + 1e-10)) / np.log10(14)
    chop = np.where(range_14 == 0, 100, chop)  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute 1d volume average for confirmation (using 1d HTF for stability)
    volume_1d = df_1d['volume'].values
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_20_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20_1d)
    
    # Pre-compute 4h Donchian(20)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(20) for stoploss
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_20_4h = pd.Series(tr_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop_multiplier = 2.5
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr_20_4h[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(avg_volume_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average of 1d volume
        vol_spike = prices['volume'].iloc[i] > (1.5 * avg_volume_20_1d_aligned[i])
        
        # Regime filter: CHOP < 38.2 = trending (favor breakouts), CHOP > 61.8 = ranging (avoid)
        is_trending = chop_aligned[i] < 38.2
        is_ranging = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower (breakdown) or ATR stop hit
            if prices['close'].iloc[i] < donchian_lower[i] or \
               prices['close'].iloc[i] < entry_price - atr_stop_multiplier * atr_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper (breakout) or ATR stop hit
            if prices['close'].iloc[i] > donchian_upper[i] or \
               prices['close'].iloc[i] > entry_price + atr_stop_multiplier * atr_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume and regime filters
            if vol_spike and is_trending:
                # Breakout long: price closes above Donchian upper
                if prices['close'].iloc[i] > donchian_upper[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Breakout short: price closes below Donchian lower
                elif prices['close'].iloc[i] < donchian_lower[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals