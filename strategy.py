#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and 1d chop regime filter
# - Long when price breaks above Camarilla H3 level AND volume > 1.5x 20-period average AND 1d chop < 38.2 (trending)
# - Short when price breaks below Camarilla L3 level AND volume > 1.5x 20-period average AND 1d chop < 38.2 (trending)
# - Exit when price returns to Camarilla Pivot level (mean reversion to equilibrium)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivot levels provide intraday support/resistance derived from prior day's range
# - Volume confirmation ensures breakouts have institutional participation
# - Chop filter ensures we only trade in trending conditions where breakouts work

name = "12h_1d_camarilla_breakout_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Pre-compute 12h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d Camarilla pivot levels from prior day
    # Camarilla levels based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels for each day
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    H3 = pivot + (range_1d * 1.1 / 4)
    L3 = pivot - (range_1d * 1.1 / 4)
    H4 = pivot + (range_1d * 1.1 / 2)
    L4 = pivot - (range_1d * 1.1 / 2)
    
    # Align HTF indicators to 12h timeframe (use previous day's levels)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Pre-compute 1d chop regime (choppiness index)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])
    tr3 = np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first element is NaN
    
    # ATR(14)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    
    # Chop = 100 * log10(tr_sum / range_max_min) / log10(14)
    chop = 100 * np.log10(tr_sum / range_max_min) / np.log10(14)
    chop = np.concatenate([np.full(13, np.nan), chop[13:]])  # align indices
    
    # Chop regime: < 38.2 = trending (good for breakout strategies)
    chop_trending = chop < 38.2
    
    # Align HTF chop regime to 12h timeframe
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop_trending_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND chop trending
            if (close[i] > H3_aligned[i] and 
                volume_spike[i] and 
                chop_trending_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND chop trending
            elif (close[i] < L3_aligned[i] and 
                  volume_spike[i] and 
                  chop_trending_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to pivot level (mean reversion to equilibrium)
            exit_long = (position == 1 and 
                        close[i] < pivot_aligned[i])
            exit_short = (position == -1 and 
                         close[i] > pivot_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals