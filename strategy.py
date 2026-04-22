#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Williams %R mean reversion.
# Choppiness Index > 61.8 indicates ranging market (mean reversion regime).
# Williams %R < -80 = oversold, > -20 = overbought. Enter mean reversion trades in ranging markets.
# Exit when Williams %R reverts to -50 (mean) or Choppiness drops below 38.2 (trending regime).
# Designed for low trade frequency (~20-40/year) to minimize fee decay in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Williams %R calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R on daily data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 14-period Choppiness Index on daily data
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low since no previous close
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr * 14 / (max_high - min_low)) / np.log10(14)
    # Handle division by zero when max_high == min_low
    chop = np.where((max_high - min_low) == 0, 50, chop)
    
    # Align 1d indicators to 4h timeframe (waits for 1d bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        wr_val = williams_r_aligned[i]
        
        # Regime filter: only trade in ranging markets (Choppiness > 61.8)
        ranging_market = chop_val > 61.8
        
        if position == 0:
            # Enter long when oversold in ranging market
            if ranging_market and wr_val < -80:
                signals[i] = 0.25
                position = 1
            # Enter short when overbought in ranging market
            elif ranging_market and wr_val > -20:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R reverts to mean (-50) or market starts trending
                if wr_val >= -50 or chop_val < 38.2:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R reverts to mean (-50) or market starts trending
                if wr_val <= -50 or chop_val < 38.2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Chop_WilliamsR_MeanReversion"
timeframe = "4h"
leverage = 1.0