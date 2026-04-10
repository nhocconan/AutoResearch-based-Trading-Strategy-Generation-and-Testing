#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d regime filter and volume confirmation
# - Long when Alligator jaws (13-period SMMA) < teeth (8-period SMMA) < lips (5-period SMMA) AND price > lips
# - Short when Alligator jaws > teeth > lips AND price < lips
# - Filter: Only trade when 1d choppiness index > 61.8 (ranging market) for mean reversion edge
# - Volume confirmation: volume > 1.5x 20-period average
# - Exit when Alligator lines re-cross (jaws/teeth/lips lose alignment)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Williams Alligator identifies trending vs ranging markets with clear entry/exit rules
# - Choppiness filter ensures we trade in ranging conditions where Alligator works best
# - Volume confirmation reduces false signals

name = "4h_1d_alligator_chop_volume_v1"
timeframe = "4h"
leverage = 1.0

def smma(arr, period):
    """Smoothed Moving Average (Williams Alligator)"""
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return result
    # First value is simple average
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute Williams Alligator (5,8,13 SMMA of median price)
    median_price = (high + low) / 2
    lips = smma(median_price, 5)   # 5-period SMMA
    teeth = smma(median_price, 8)  # 8-period SMMA
    jaws = smma(median_price, 13)  # 13-period SMMA
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d Choppiness Index for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - using Wilder's smoothing
    atr_1d = np.zeros_like(tr)
    if len(tr) >= 14:
        atr_1d[13] = np.mean(tr[1:14])  # First ATR value
        for i in range(14, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over last 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over last 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop = np.full_like(close_1d, 50.0, dtype=float)  # Default to neutral
    mask = (range_14 > 0) & (~np.isnan(sum_atr_14)) & (sum_atr_14 > 0)
    chop[mask] = 100 * np.log10(sum_atr_14[mask] / range_14[mask]) / np.log10(14)
    
    # Regime: CHOP > 61.8 = ranging market (good for Alligator mean reversion)
    ranging_regime = chop > 61.8
    
    # Align HTF indicators to 4h timeframe
    ranging_regime_aligned = align_htf_to_ltf(prices, df_1d, ranging_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaws[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ranging_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator aligned for uptrend: jaws < teeth < lips
            alligator_long = jaws[i] < teeth[i] < lips[i]
            # Alligator aligned for downtrend: jaws > teeth > lips
            alligator_short = jaws[i] > teeth[i] > lips[i]
            
            # Long conditions: Alligator uptrend AND price > lips AND ranging regime AND volume spike
            if (alligator_long and close[i] > lips[i] and 
                ranging_regime_aligned[i] and volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Alligator downtrend AND price < lips AND ranging regime AND volume spike
            elif (alligator_short and close[i] < lips[i] and 
                  ranging_regime_aligned[i] and volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Alligator alignment breaks (jaws/teeth/lips lose alignment)
            alligator_aligned = ((jaws[i] < teeth[i] < lips[i]) or 
                                (jaws[i] > teeth[i] > lips[i]))
            
            if not alligator_aligned:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals