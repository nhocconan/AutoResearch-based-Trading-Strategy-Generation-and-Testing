#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 1d Volume Spike and Chop Regime Filter
# Williams %R identifies overbought/oversold conditions; extreme readings (>90 or <10) 
# combined with 1d volume spike suggest exhaustion and potential reversal.
# Chop regime filter (CHOP > 61.8) ensures we only trade in ranging markets where 
# mean reversion works best, avoiding whipsaws in strong trends.
# Designed for 6h timeframe to capture medium-term reversals with low trade frequency.
# Target: 12-37 trades/year (50-150 over 4 years) with discrete position sizing.

name = "6h_WilliamsR_Extreme_1dVolumeSpike_ChopRegime"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Williams %R, volume, and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (%R = (Highest High - Close) / (Highest High - Lowest Low) * -100)
    # Using 14-period lookback
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high_14 - lowest_low_14
    williams_r = np.where(rr != 0, ((highest_high_14 - df_1d['close'].values) / rr) * -100, -50)
    
    # Extreme levels: >90 (oversold) or <10 (overbought)
    williams_extreme = (williams_r > 90) | (williams_r < 10)
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Calculate 1d Chopiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR1) / (n * log(n))) / log10(n)
    # Using 14-period CHOP
    tr1 = np.maximum(df_1d['high'].values - df_1d['low'].values,
                     np.maximum(np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
                                np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))))
    # Handle first bar
    tr1[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    n_val = 14
    chop = np.where((sum_atr1 > 0) & (n_val > 1),
                    100 * np.log10(sum_atr1 / (n_val * np.log(n_val))) / np.log10(n_val),
                    50)  # default to neutral
    chop_regime = chop > 61.8  # ranging market
    
    # Align 1d indicators to 6h timeframe
    williams_extreme_aligned = align_htf_to_ltf(prices, df_1d, williams_extreme)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if any value is NaN or outside session
        if (not williams_extreme_aligned[i] or 
            np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_regime_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < 10 (oversold) with volume spike in ranging market
            if williams_r[-1] < 10 and volume_spike_aligned[i] and chop_regime_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > 90 (overbought) with volume spike in ranging market
            elif williams_r[-1] > 90 and volume_spike_aligned[i] and chop_regime_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -10 (exit oversold) or opposite extreme
            if williams_r[-1] > -10 or williams_r[-1] > 90:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < 90 (exit overbought) or opposite extreme
            if williams_r[-1] < 90 or williams_r[-1] < 10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals