# 4h_Camarilla_R1_S1_Breakout_Volume_Regime_v1
# Camarilla pivot levels from 1d + volume spike + choppiness regime filter
# Uses 1d Camarilla levels as support/resistance with volume confirmation and chop filter to avoid whipsaws
# Works in both bull and bear markets by combining mean reversion at pivot levels with trend filtering
# Target: 75-200 total trades over 4 years (19-50/year)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Camarilla pivot levels (R1, S1) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        high_low = high_1d[i] - low_1d[i]
        camarilla_r1[i] = close_1d[i] + high_low * 1.1 / 12
        camarilla_s1[i] = close_1d[i] - high_low * 1.1 / 12
    
    # === 1d Choppiness Index (14-period) for regime filter ===
    # Calculate True Range and directional movement
    tr = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM over 14 periods
    atr_14 = np.full_like(tr, np.nan)
    plus_dm_14 = np.full_like(plus_dm, np.nan)
    minus_dm_14 = np.full_like(minus_dm, np.nan)
    
    for i in range(len(tr)):
        if i >= 14:
            if i == 14:
                atr_14[i] = np.nansum(tr[1:i+1])
                plus_dm_14[i] = np.nansum(plus_dm[1:i+1])
                minus_dm_14[i] = np.nansum(minus_dm[1:i+1])
            else:
                atr_14[i] = atr_14[i-1] - (atr_14[i-1] / 14) + tr[i]
                plus_dm_14[i] = plus_dm_14[i-1] - (plus_dm_14[i-1] / 14) + plus_dm[i]
                minus_dm_14[i] = minus_dm_14[i-1] - (minus_dm_14[i-1] / 14) + minus_dm[i]
        elif i > 0:
            atr_14[i] = np.nanmean(tr[1:i+1]) if np.sum(~np.isnan(tr[1:i+1])) > 0 else np.nan
            plus_dm_14[i] = np.nanmean(plus_dm[1:i+1]) if np.sum(~np.isnan(plus_dm[1:i+1])) > 0 else np.nan
            minus_dm_14[i] = np.nanmean(minus_dm[1:i+1]) if np.sum(~np.isnan(minus_dm[1:i+1])) > 0 else np.nan
    
    # Calculate Chop = 100 * log10(sum(TR14) / (sum(+DM14) + sum(-DM14))) / log10(14)
    chop = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i >= 14 and not (np.isnan(plus_dm_14[i]) or np.isnan(minus_dm_14[i]) or np.isnan(atr_14[i])):
            if plus_dm_14[i] + minus_dm_14[i] > 0:
                chop[i] = 100 * np.log10(atr_14[i] / (plus_dm_14[i] + minus_dm_14[i])) / np.log10(14)
            else:
                chop[i] = 50
    
    # === Align indicators to 4h timeframe ===
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    volume_spike = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Chop regime: chop > 50 indicates ranging market (good for mean reversion at pivots)
        chop_regime = chop_aligned[i] > 50
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price touches or goes below S1 with volume spike in ranging market
            if (low[i] <= camarilla_s1_aligned[i] and 
                volume_spike[i] and 
                chop_regime):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price touches or goes above R1 with volume spike in ranging market
            elif (high[i] >= camarilla_r1_aligned[i] and 
                  volume_spike[i] and 
                  chop_regime):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price reaches or exceeds R1 OR chop drops below 40 (trending)
            if (high[i] >= camarilla_r1_aligned[i] or 
                chop_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches or goes below S1 OR chop drops below 40 (trending)
            if (low[i] <= camarilla_s1_aligned[i] or 
                chop_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0