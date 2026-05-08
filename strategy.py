#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Choppiness Index regime filter with weekly trend filter and volume confirmation
# We go long when price closes above the upper Donchian channel (20) in trending markets (CHOP < 38.2) with weekly EMA(34) uptrend and volume spike.
# We go short when price closes below the lower Donchian channel (20) in trending markets (CHOP < 38.2) with weekly EMA(34) downtrend and volume spike.
# In ranging markets (CHOP >= 38.2), we mean-revert at Donchian channel boundaries with volume confirmation.
# Uses 1d timeframe to target 7-25 trades/year, avoiding excessive frequency.
# Choppiness Index filters market regime to avoid whipsaws in sideways markets.
# Weekly trend filter ensures we trade with the higher timeframe momentum.
# Volume spike confirms institutional participation.

name = "1d_ChopRegime_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for Donchian channels and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 days for Donchian and Chop
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index (14-period)
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low over 14))
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close[:-1])
    tr3 = np.abs(low_1d[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    chop = np.full_like(close, np.nan)
    mask = (range_hl > 0) & (~np.isnan(atr))
    chop[mask] = 100 * np.log10(np.sum(atr) / np.log10(range_hl[mask])) if np.sum(atr) > 0 else np.nan
    # Correct calculation: Chop = 100 * log10(sum(ATR14) / log10(HH14 - LL14))
    sum_atr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_minus_ll = highest_high - lowest_low
    chop = np.full_like(close, np.nan)
    valid = (hh_minus_ll > 0) & (~np.isnan(sum_atr))
    chop[valid] = 100 * np.log10(sum_atr[valid] / np.log10(hh_minus_ll[valid]))
    
    # Align Donchian and Chop to lower timeframe (they are already daily)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Trending market: CHOP < 38.2
            if chop_val < 38.2:
                # Enter long: price breaks above Donchian high + weekly uptrend + volume spike
                if close[i] > donchian_high_val and close[i] > ema34_1w_val and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Enter short: price breaks below Donchian low + weekly downtrend + volume spike
                elif close[i] < donchian_low_val and close[i] < ema34_1w_val and vol_spike:
                    signals[i] = -0.25
                    position = -1
            # Ranging market: CHOP >= 38.2
            else:
                # Enter long: price at Donchian low + volume spike (mean reversion)
                if close[i] <= donchian_low_val * 1.001 and vol_spike:  # Allow small tolerance
                    signals[i] = 0.25
                    position = 1
                # Enter short: price at Donchian high + volume spike (mean reversion)
                elif close[i] >= donchian_high_val * 0.999 and vol_spike:  # Allow small tolerance
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR weekly trend turns down
            if close[i] < donchian_low_val or close[i] < ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR weekly trend turns up
            if close[i] > donchian_high_val or close[i] > ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals