#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and chop regime filter
# - Long when price breaks above 4h Donchian(20) high AND 1d volume > 1.5x 20-day average AND 1d chop > 61.8 (ranging)
# - Short when price breaks below 4h Donchian(20) low AND 1d volume > 1.5x 20-day average AND 1d chop > 61.8 (ranging)
# - Exit when price returns to 4h Donchian(20) midpoint
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian breakouts work in ranging markets (common in 2025+) when confirmed with volume and chop filter
# - Volume spike confirms breakout validity, chop filter ensures ranging regime
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(n)
    
    # Pre-compute 4h Donchian Channel (20-period)
    def highest_high(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def lowest_low(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_high = highest_high(high, 20)
    donchian_low = lowest_low(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_avg_1d = rolling_mean(volume_1d, 20)
    vol_ratio_1d = np.zeros_like(volume_1d)
    for i in range(len(volume_1d)):
        if vol_avg_1d[i] > 0:
            vol_ratio_1d[i] = volume_1d[i] / vol_avg_1d[i]
        else:
            vol_ratio_1d[i] = 1.0
    
    # Pre-compute 1d Choppiness Index (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    def rolling_sum(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.sum(arr[i - window + 1:i + 1])
        return result
    
    # Calculate 1d True Range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.zeros_like(high_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr_1d[i] = true_range(high_1d[i], low_1d[i], close_1d[i-1])
    
    # Calculate 1d ATR (14-period) for chop calculation
    atr_1d = np.zeros_like(tr_1d)
    atr_1d[13] = np.mean(tr_1d[1:15])
    for i in range(14, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate 1d Choppiness Index
    hh_1d = highest_high(high_1d, 14)
    ll_1d = lowest_low(low_1d, 14)
    chop_1d = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if hh_1d[i] > ll_1d[i]:
            log_sum = np.log10(rolling_sum(tr_1d, 14)[i] / (hh_1d[i] - ll_1d[i]))
            chop_1d[i] = 100 * log_sum / np.log10(14)
        else:
            chop_1d[i] = 50.0
    
    chop_regime_1d = chop_1d > 61.8  # Ranging market
    vol_spike_1d = vol_ratio_1d > 1.5  # Volume spike
    
    # Align HTF indicators to 4h timeframe
    chop_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_regime_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Donchian breakout up + volume spike + chop regime
            if (close[i] > donchian_high[i] and 
                vol_spike_1d_aligned[i] and 
                chop_regime_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Donchian breakout down + volume spike + chop regime
            elif (close[i] < donchian_low[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_regime_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Donchian midpoint
            exit_long = (position == 1 and close[i] <= donchian_mid[i])
            exit_short = (position == -1 and close[i] >= donchian_mid[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals