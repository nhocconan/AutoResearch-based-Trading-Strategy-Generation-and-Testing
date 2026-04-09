#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume spike and choppiness regime filter
# In ranging regimes (CHOP > 61.8): buy when Williams %R < -80 (oversold) with volume confirmation
# In trending regimes (CHOP < 38.2): buy when Williams %R crosses above -50 from below with volume confirmation
# Short symmetrically. Uses discrete position sizing 0.25 to limit trades to ~20-50/year and reduce fee drag
# Williams %R identifies overextended moves; volume confirms institutional participation; chop filter avoids whipsaws

name = "4h_1d_williamsr_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = np.where(
        (highest_high_1d - lowest_low_1d) != 0,
        (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100,
        -50  # neutral when range is zero
    )
    
    # Calculate 1d average volume (20-period)
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Align 1d indicators to 4h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute volume confirmation array
    volume_confirmed = volume > 1.5 * avg_volume_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if Williams %R rises above -20 (overbought) or we enter ranging regime
                if williams_r_1d_aligned[i] > -20 or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if Williams %R rises above -50 or price rises above recent high
                if williams_r_1d_aligned[i] > -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if Williams %R falls below -80 (oversold) or we enter ranging regime
                if williams_r_1d_aligned[i] < -80 or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if Williams %R falls below -50 or price drops below recent low
                if williams_r_1d_aligned[i] < -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Enter long when Williams %R crosses above -50 from below with volume confirmation
                if (williams_r_1d_aligned[i] > -50 and 
                    i > 0 and williams_r_1d_aligned[i-1] <= -50 and 
                    volume_confirmed[i]):
                    position = 1
                    signals[i] = 0.25
                # Enter short when Williams %R crosses below -50 from above with volume confirmation
                elif (williams_r_1d_aligned[i] < -50 and 
                      i > 0 and williams_r_1d_aligned[i-1] >= -50 and 
                      volume_confirmed[i]):
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion: buy oversold, sell overbought
                if williams_r_1d_aligned[i] < -80 and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                elif williams_r_1d_aligned[i] > -20 and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals