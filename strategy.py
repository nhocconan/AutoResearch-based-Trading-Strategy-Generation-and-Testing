#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1d Choppiness Index regime filter with 1d Donchian(20) breakout and volume confirmation.
# In choppy markets (CHOP > 61.8), we mean-revert at Donchian boundaries; in trending markets (CHOP < 38.2), we follow breakouts.
# Uses 1d structure for regime and trend, 4h for entry timing and volume confirmation.
# Designed for low trade frequency (15-25/year) to avoid fee drag. Works in bull/bear markets via regime adaptation.

name = "4h_1dChop_Donchian20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Choppiness Index (14-period)
    def true_range(high, low, close_prev):
        tr = np.maximum(high - low, 
                        np.maximum(np.abs(high - close_prev), 
                                   np.abs(low - close_prev)))
        return tr
    
    close_prev_1d = np.roll(close_1d, 1)
    close_prev_1d[0] = close_1d[0]
    tr_1d = true_range(high_1d, low_1d, close_prev_1d)
    
    atr_14 = np.zeros_like(tr_1d)
    for i in range(len(tr_1d)):
        if i < 14:
            atr_14[i] = np.nan
        else:
            atr_14[i] = np.mean(tr_1d[i-13:i+1])
    
    sum_tr_14 = np.zeros_like(tr_1d)
    max_hh = np.zeros_like(high_1d)
    min_ll = np.zeros_like(low_1d)
    
    for i in range(len(tr_1d)):
        if i < 14:
            sum_tr_14[i] = np.nan
            max_hh[i] = np.nan
            min_ll[i] = np.nan
        else:
            sum_tr_14[i] = np.sum(tr_1d[i-13:i+1])
            max_hh[i] = np.max(high_1d[i-13:i+1])
            min_ll[i] = np.min(low_1d[i-13:i+1])
    
    chop = 100 * np.log10(sum_tr_14 / (max_hh - min_ll)) / np.log10(14)
    
    # Calculate 1d Donchian channel (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    donchian_high_1d = rolling_max(high_1d, 20)
    donchian_low_1d = rolling_min(low_1d, 20)
    
    # Align 1d indicators to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # Volume spike: 1.5x 20-period EMA
    vol_ema = pd.Series(volume_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume_4h > (vol_ema * 1.5)
    
    # Align volume spike to 4h timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_4h, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        upper_bound = donchian_high_aligned[i]
        lower_bound = donchian_low_aligned[i]
        
        if position == 0:
            # Determine regime and enter accordingly
            if chop_val > 61.8:  # Choppy market - mean reversion
                # Enter long near lower Donchian boundary with volume spike
                if close[i] <= lower_bound * 1.005 and vol_spike_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Enter short near upper Donchian boundary with volume spike
                elif close[i] >= upper_bound * 0.995 and vol_spike_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif chop_val < 38.2:  # Trending market - follow breakout
                # Enter long on upward breakout with volume spike
                if close[i] > upper_bound and vol_spike_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Enter short on downward breakout with volume spike
                elif close[i] < lower_bound and vol_spike_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            # In transition zone (38.2 <= CHOP <= 61.8), wait for clearer signal
        elif position == 1:
            # Exit long: price reaches opposite Donchian boundary or loses volume momentum
            if close[i] >= upper_bound:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches opposite Donchian boundary or loses volume momentum
            if close[i] <= lower_bound:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals