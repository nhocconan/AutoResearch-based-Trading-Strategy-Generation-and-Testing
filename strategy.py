#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2.0x 20-period average AND choppiness index > 61.8 (range regime)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2.0x 20-period average AND choppiness index > 61.8
# - Exit when price returns to Camarilla Pivot level (mean reversion to center)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Camarilla pivots provide precise intraday support/resistance levels
# - Volume confirmation ensures breakout validity
# - Choppiness filter ensures we trade in ranging markets where mean reversion works

name = "4h_1d_camarilla_breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Pre-compute 1d choppiness index (14-period)
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
    
    # Choppiness Index = 100 * log10(sum(TR,14) / (max(HH,14) - min(LL,14))) / log10(14)
    def rolling_sum(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.sum(arr[i - window + 1:i + 1])
        return result
    
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    tr_sum = rolling_sum(tr, 14)
    hh = rolling_max(high_1d, 14)
    ll = rolling_min(low_1d, 14)
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    chop[hh - ll == 0] = 0  # Avoid division by zero
    chop_regime = chop > 61.8  # Range regime
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day)
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.25 * (High - Low)
    # H2 = Close + 1.166 * (High - Low)
    # H1 = Close + 0.833 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    # L1 = Close - 0.833 * (High - Low)
    # L2 = Close - 1.166 * (High - Low)
    # L3 = Close - 1.25 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    
    # First day has no previous data
    high_prev[0] = high_1d[0]
    low_prev[0] = low_1d[0]
    close_prev[0] = close_1d[0]
    
    pivot = (high_prev + low_prev + close_prev) / 3
    range_val = high_prev - low_prev
    
    h3 = close_prev + 1.25 * range_val
    l3 = close_prev - 1.25 * range_val
    
    # Align HTF indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(volume_spike_aligned[i]) or np.isnan(chop_regime_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pivot_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND chop regime (range)
            if (close[i] > h3_aligned[i] and 
                volume_spike_aligned[i] and 
                chop_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND chop regime (range)
            elif (close[i] < l3_aligned[i] and 
                  volume_spike_aligned[i] and 
                  chop_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to pivot level (mean reversion)
            exit_long = (position == 1 and close[i] < pivot_aligned[i])
            exit_short = (position == -1 and close[i] > pivot_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals