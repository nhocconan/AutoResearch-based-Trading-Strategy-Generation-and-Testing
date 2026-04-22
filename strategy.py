#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 1d Williams %R mean reversion.
# Choppiness Index > 61.8 indicates ranging market (mean reversion regime), < 38.2 indicates trending.
# In ranging markets, Williams %R > -20 (overbought) triggers short, < -80 (oversold) triggers long.
# Uses 1d Williams %R for higher timeframe signal quality, reducing false signals in 4h.
# Designed for low trade frequency (~20-40/year) to minimize fee decay.
# Works in both bull and bear markets by adapting to market regime.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Williams %R calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R on 1d data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_1d) / (highest_high - lowest_low)) * -100
    
    # Calculate 14-period Choppiness Index on 1d data
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    # Align 1d indicators to 4h timeframe (waits for 1d bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        chop_val = chop_aligned[i]
        wr_val = williams_r_aligned[i]
        
        # Regime filter: Chop > 61.8 = ranging (mean reversion), Chop < 38.2 = trending
        # We only trade in ranging markets for mean reversion
        if chop_val > 61.8:
            # Mean reversion signals in ranging market
            if position == 0:
                # Long when oversold (Williams %R < -80)
                if wr_val < -80:
                    signals[i] = 0.25
                    position = 1
                # Short when overbought (Williams %R > -20)
                elif wr_val > -20:
                    signals[i] = -0.25
                    position = -1
            
            elif position != 0:
                # Exit when Williams %R returns to neutral zone (-50 to -50)
                exit_signal = False
                
                if position == 1:  # long position
                    # Exit when Williams %R rises above -50 (mean reversion complete)
                    if wr_val > -50:
                        exit_signal = True
                
                elif position == -1:  # short position
                    # Exit when Williams %R falls below -50 (mean reversion complete)
                    if wr_val < -50:
                        exit_signal = True
                
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    # Hold position
                    signals[i] = 0.25 if position == 1 else -0.25
        else:
            # In trending market, stay flat to avoid whipsaw
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Chop_WilliamsR_MeanReversion"
timeframe = "4h"
leverage = 1.0