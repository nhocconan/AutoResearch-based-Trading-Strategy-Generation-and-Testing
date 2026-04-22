#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Williams %R mean reversion.
# Uses daily Williams %R for mean reversion signals (oversold/overbought).
# Enters only when 4h Choppiness > 61.8 (ranging market) to avoid trending whipsaws.
# Long when Williams %R < -80 (oversold) in ranging market.
# Short when Williams %R > -20 (overbought) in ranging market.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Designed for low trade frequency (<30/year) to minimize fee drag in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data for Williams %R (overbought/oversold)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R (14-period): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Load 4h data for Choppiness Index (regime filter)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range and ATR(14) for Choppiness
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    
    # Align Williams %R to 4h timeframe (waits for 1d bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in ranging markets (Choppiness > 61.8)
        if chop[i] > 61.8:
            if position == 0:
                # Long: Williams %R oversold (< -80)
                if williams_r_aligned[i] < -80:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R overbought (> -20)
                elif williams_r_aligned[i] > -20:
                    signals[i] = -0.25
                    position = -1
            else:
                # Exit: Williams %R crosses midpoint (-50)
                if position == 1 and williams_r_aligned[i] > -50:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and williams_r_aligned[i] < -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
        else:
            # In trending markets, stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Chop_WilliamsR_MeanReversion"
timeframe = "4h"
leverage = 1.0