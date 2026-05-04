#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume spike and chop regime filter
# In trending markets (1d CHOP < 42): trade in direction of Alligator alignment (jaw-teeth-lips)
# In ranging markets (1d CHOP >= 42): fade extreme deviations from Alligator mean (mean reversion)
# Volume confirmation (>1.3x 20-period EMA) filters low-quality signals
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 trades over 4 years.
# Williams Alligator uses SMAs of median price: Jaw=13, Teeth=8, Lips=5
# Strategy adapts to bull/bear markets via regime filter and uses 12h primary timeframe.

name = "12h_WilliamsAlligator_1dChop_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (CHOP) - 14 period
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # True Range
    tr1 = high_1d.sub(low_1d)
    tr2 = high_1d.sub(close_1d.shift(1)).abs()
    tr3 = low_1d.sub(close_1d.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over 14 periods
    tr_sum_14 = tr.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh_14 = high_1d.rolling(window=14, min_periods=14).max()
    ll_14 = low_1d.rolling(window=14, min_periods=14).min()
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    hh_ll_diff = hh_14 - ll_14
    chop_1d = np.where(
        (hh_ll_diff > 0) & (~tr_sum_14.isna()) & (~hh_ll_diff.isna()),
        100 * np.log10(tr_sum_14 / hh_ll_diff) / np.log10(14),
        50.0  # neutral when undefined
    )
    
    # Align 1d chop to 12h timeframe (completed 1d bar only)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Williams Alligator on 12h timeframe
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw (13-period SMA), Teeth (8-period SMA), Lips (5-period SMA)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Alligator alignment: 
    # Trending up: Lips > Teeth > Jaw
    # Trending down: Lips < Teeth < Jaw
    # We'll use the deviation from the mean as signal
    alligator_mean = (jaw + teeth + lips) / 3
    deviation = median_price - alligator_mean
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            if chop_aligned[i] < 42:
                # Trending market: trade in direction of Alligator alignment
                if lips[i] > teeth[i] > jaw[i]:
                    # Uptrend: long on positive deviation
                    if deviation[i] > 0 and volume_confirm:
                        signals[i] = 0.25
                        position = 1
                elif lips[i] < teeth[i] < jaw[i]:
                    # Downtrend: short on negative deviation
                    if deviation[i] < 0 and volume_confirm:
                        signals[i] = -0.25
                        position = -1
            else:
                # Ranging market: fade extreme deviations (mean reversion)
                # Use 2-standard deviation bands
                if i >= 20:
                    dev_mean = np.nanmean(deviation[max(0, i-20):i])
                    dev_std = np.nanstd(deviation[max(0, i-20):i])
                    if not np.isnan(dev_std) and dev_std > 0:
                        upper_band = dev_mean + 2.0 * dev_std
                        lower_band = dev_mean - 2.0 * dev_std
                        
                        if deviation[i] <= lower_band and volume_confirm:
                            # Long at lower band
                            signals[i] = 0.25
                            position = 1
                        elif deviation[i] >= upper_band and volume_confirm:
                            # Short at upper band
                            signals[i] = -0.25
                            position = -1
        elif position == 1:
            # Exit long: deviation returns to zero OR chop increases (>50) OR volume drops
            if (deviation[i] <= 0 or 
                chop_aligned[i] > 50 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: deviation returns to zero OR chop increases (>50) OR volume drops
            if (deviation[i] >= 0 or 
                chop_aligned[i] > 50 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals