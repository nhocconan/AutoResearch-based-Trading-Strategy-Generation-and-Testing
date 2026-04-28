#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Weekly pivot (PP) from prior week determines bias: price > PP = long bias, price < PP = short bias.
# Breakout above Donchian(20) high with long bias and volume spike = long.
# Breakdown below Donchian(20) low with short bias and volume spike = short.
# Exit on retracement to midpoint of Donchian channel.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).
# Weekly pivot provides genuine HTF edge not yet saturated in 6h timeframe.
# Volume confirmation filters weak breakouts.
# Works in both bull/bear markets by requiring alignment with weekly pivot bias.

name = "6h_Donchian20_Breakout_WeeklyPivot_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot (requires weekly OHLC)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior completed week
    # PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    prior_high = df_1w['high'].shift(1).values
    prior_low = df_1w['low'].shift(1).values
    prior_close = df_1w['close'].shift(1).values
    
    pp = (prior_high + prior_low + prior_close) / 3.0
    r1 = 2 * pp - prior_low
    s1 = 2 * pp - prior_high
    
    # Align weekly pivot levels to 6h (they change only when weekly bar closes)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Donchian(20) channel on 6h data
    lookback = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = low_series.rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # Ensure sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Weekly pivot bias
        bias_long = close[i] > pp_aligned[i]
        bias_short = close[i] < pp_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > Donchian high, long bias, volume confirm
            if price > donchian_high[i] and bias_long and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Price < Donchian low, short bias, volume confirm
            elif price < donchian_low[i] and bias_short and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement to midpoint
            if price < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on retracement to midpoint
            if price > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals