#!/usr/bin/env python3
# 4h_donchian_12h_volume_chop_v1
# Hypothesis: 4h Donchian channel breakout with 12h volume confirmation and chop regime filter.
# Uses Donchian(20) breakouts for trend capture, confirmed by 12h volume spike (>1.5x 20-period average).
# Chop regime filter (CHOP > 61.8) ensures trades only in ranging markets where mean reversion works.
# Designed for 19-50 trades/year (75-200 over 4 years) with discrete position sizing to minimize fee drag.
# Works in bull/bear markets: Donchian captures breakouts, volume confirms institutional interest,
# chop filter avoids whipsaw in strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_12h_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper/lower (20-period high/low)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (completed 4h candle only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Get 12h HTF data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h volume spike confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (vol_ma_20_12h * 1.5)
    
    # Align 12h volume spike to 4h timeframe
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h.astype(float))
    
    # Calculate Chop regime filter on 4h (true = ranging market)
    atr_period = 14
    tr1 = pd.Series(high[1:]).values - pd.Series(low[1:]).values
    tr2 = np.abs(pd.Series(high[1:]).values - pd.Series(close[:-1]).values)
    tr3 = np.abs(pd.Series(low[1:]).values - pd.Series(close[:-1]).values)
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Chop = 100 * log15(sum(ATR14) / (max(high)-min(low))) / log15(14)
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    max_min_diff = max_high - min_low
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero and handle NaN
    chop_raw = np.where((max_min_diff > 0) & (~np.isnan(sum_atr)), 
                        100 * np.log10(sum_atr / max_min_diff) / np.log10(14), 
                        np.nan)
    chop = pd.Series(chop_raw).rolling(window=1, min_periods=1).mean().values  # just propagate
    
    # Chop > 61.8 indicates ranging market (good for mean reversion/breakout fade)
    chop_ranging = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_spike_12h_aligned[i]) or np.isnan(chop_ranging[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or chop breaks down (trending)
            if (close[i] < donchian_lower_aligned[i]) or (not chop_ranging[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or chop breaks down (trending)
            if (close[i] > donchian_upper_aligned[i]) or (not chop_ranging[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above Donchian upper, with 12h volume spike, in ranging market
            if (close[i] > donchian_upper_aligned[i]) and vol_spike_12h_aligned[i] and chop_ranging[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian lower, with 12h volume spike, in ranging market
            elif (close[i] < donchian_lower_aligned[i]) and vol_spike_12h_aligned[i] and chop_ranging[i]:
                position = -1
                signals[i] = -0.25
    
    return signals