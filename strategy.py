#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter
# Donchian breakout captures institutional accumulation/distribution
# 1d volume spike > 2x 20-period EMA confirms participation
# Choppiness Index (14) < 38.2 ensures trending market to avoid range-bound whipsaws
# Designed for 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Works in bull markets (breakout above upper band + chop < 38.2) and bear markets (breakout below lower band + chop < 38.2)

name = "4h_Donchian20_VolumeSpike_ChopFilter"
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
    
    # 1d data for volume spike and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for calculations
        return np.zeros(n)
    
    # 1d Volume EMA(20) for spike detection
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # 1d Choppiness Index (14)
    # TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = abs(df_1d['low'] - df_1d['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh_14 = df_1d['high'].rolling(window=14, min_periods=14).max()
    ll_14 = df_1d['low'].rolling(window=14, min_periods=14).min()
    
    # Chop = 100 * log10(atr_14 / (hh_14 - ll_14)) / log10(14)
    chop = 100 * np.log10(atr_14 / (hh_14 - ll_14)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: trending market (chop < 38.2)
        trending_market = chop_aligned[i] < 38.2
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper Donchian band with volume spike and trending market
            if close[i] > high_20[i] and volume_spike_aligned[i] and trending_market:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below lower Donchian band with volume spike and trending market
            elif close[i] < low_20[i] and volume_spike_aligned[i] and trending_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below lower Donchian band (reversal) OR market becomes choppy
            if close[i] < low_20[i] or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian band (reversal) OR market becomes choppy
            if close[i] > high_20[i] or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals