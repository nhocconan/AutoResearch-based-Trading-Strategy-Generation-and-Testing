#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 20-period average AND choppy regime (CHOP > 61.8)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 20-period average AND choppy regime (CHOP > 61.8)
# - Exit when price reverts to Camarilla PIVOT level (mean reversion in choppy markets)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Camarilla pivots work well in ranging/choppy markets which are common in bear phases
# - Volume confirmation ensures breakouts have conviction
# - Chop filter avoids false signals in strong trends

name = "4h_1d_camarilla_volume_chop_v1"
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
    
    # Pre-compute 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > vol_ma  # Volume above average
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations: based on previous day's range
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    H3_1d = pivot_1d + (range_1d * 1.1 / 4)
    L3_1d = pivot_1d - (range_1d * 1.1 / 4)
    H4_1d = pivot_1d + (range_1d * 1.1 / 2)
    L4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align HTF Camarilla levels to 4h timeframe
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    H4_1d_aligned = align_htf_to_ltf(prices, df_1d, H4_1d)
    L4_1d_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Pre-compute 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) / (max(high,n) - min(low,n)))
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    # First TR is high-low
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], tr_1d])
    
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Avoid division by zero
    chop_1d = np.where(
        range_14 > 0,
        100 * np.log10(sum_atr_14 / np.log10(14)) / np.log10(range_14),
        50  # Neutral when range is zero
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    choppy_regime = chop_1d_aligned > 61.8  # Choppy market regime
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(choppy_regime[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND choppy regime
            if (close[i] > H3_1d_aligned[i] and 
                volume_spike[i] and 
                choppy_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND choppy regime
            elif (close[i] < L3_1d_aligned[i] and 
                  volume_spike[i] and 
                  choppy_regime[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot (mean reversion)
            # Exit when price returns to pivot level (mean reversion in choppy markets)
            exit_long = (position == 1 and close[i] <= pivot_1d_aligned[i])
            exit_short = (position == -1 and close[i] >= pivot_1d_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals