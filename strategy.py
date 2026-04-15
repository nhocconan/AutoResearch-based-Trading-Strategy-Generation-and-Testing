#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R mean reversion with volume spike and chop regime filter.
# Williams %R < -80 = oversold (long), > -20 = overbought (short) on 1d timeframe.
# Volume confirmation: current volume > 1.5x 24-period volume SMA.
# Chop regime: only trade when market is ranging (CHOP > 61.8) to avoid whipsaws in strong trends.
# Designed for low trade frequency (15-25/year) to minimize fee drag. Works in bull/bear markets:
# - In bull: mean reversion during pullbacks in uptrend
# - In bear: mean reversion during bounces in downtrend
# - In range: classic mean reversion at extremes

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Indicators: Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    denominator = highest_high_1d - lowest_low_1d
    williams_r_1d = np.where(denominator != 0, 
                            ((highest_high_1d - df_1d['close'].values) / denominator) * -100, 
                            -50.0)  # neutral when no range
    
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # === 1d Indicators: Chopiness Index (14-period) ===
    # Chop = 100 * log10(sum(ATR) / (log10(highest high - lowest low) * sqrt(period)))
    atr_1d = np.maximum(np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
        np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    ))
    atr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # first period
    
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    max_min_range = highest_high_14 - lowest_low_14
    
    chop_1d = np.where(max_min_range != 0,
                      100 * np.log10(sum_atr_14 / (np.log10(14) * max_min_range)),
                      50.0)  # neutral when no range
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 24-period volume SMA
        vol_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        vol_confirm = volume[i] > (vol_sma_24[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(vol_sma_24[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
        if chop_1d_aligned[i] <= 61.8:
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R below -80 (oversold)
        # 2. Volume confirmation
        if (williams_r_1d_aligned[i] < -80.0 and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R above -20 (overbought)
        # 2. Volume confirmation
        elif (williams_r_1d_aligned[i] > -20.0 and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsR_MeanRev_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0