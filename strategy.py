#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d volume spike and chop regime filter
# - Long: Williams %R(14) < -80 (oversold) + 1d volume > 1.8x 20-period MA + 1d chop > 61.8 (range regime)
# - Short: Williams %R(14) > -20 (overbought) + 1d volume > 1.8x 20-period MA + 1d chop > 61.8
# - Exit: Williams %R returns to -50 level OR chop < 61.8 (trending regime)
# - Position sizing: 0.25 discrete level
# - Works in bull/bear: mean reversion in ranges, volume confirms participation, chop filter avoids false signals in trends

name = "12h_1d_williamsr_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d volume MA(20) for spike detection
    volume_ma_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sumTR14 / (hh14 - ll14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_1d = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    # Handle cases where sum_tr_14 is 0
    chop_1d = np.where(np.isnan(chop_1d) | np.isinf(chop_1d), 50, chop_1d)
    
    chop_ma_10_1d = pd.Series(chop_1d).ewm(span=10, min_periods=10, adjust=False).mean().values
    chop_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_ma_10_1d)
    
    # Align 1d volume data
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(chop_ma_10_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 20-period MA
        volume_ma_20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
        vol_confirm_12h = volume[i] > volume_ma_20[i]
        
        # 1d volume spike: current volume > 1.8x 20-period MA
        vol_spike_1d = volume_1d_aligned[i] > 1.8 * volume_ma_20_1d_aligned[i]
        
        # Chop regime: CHOP > 61.8 = range regime (favor mean reversion)
        chop_regime = chop_ma_10_1d_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for Williams %R extremes
            # Long entry: Williams %R < -80 (oversold) + vol confirm + vol spike + chop regime
            if (williams_r[i] < -80 and vol_confirm_12h and 
                vol_spike_1d and chop_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + vol confirm + vol spike + chop regime
            elif (williams_r[i] > -20 and vol_confirm_12h and 
                  vol_spike_1d and chop_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R returns to -50 level OR chop < 61.8 (trending regime)
            if position == 1:  # Long position
                if williams_r[i] >= -50 or chop_ma_10_1d_aligned[i] < 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if williams_r[i] <= -50 or chop_ma_10_1d_aligned[i] < 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals