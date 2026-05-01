#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and 1w pivot direction.
# Uses 1d ATR(14) to filter for low volatility regimes (ATR < 20-period MA of ATR) for breakout validity.
# Uses 1w Camarilla pivot levels (S1/R1) for directional bias: long only above weekly S1, short only below weekly R1.
# Enter long when price breaks above 6h Donchian upper (20) in low ATR regime and above weekly S1.
# Enter short when price breaks below 6h Donchian lower (20) in low ATR regime and below weekly R1.
# Exit on opposite Donchian break or ATR expansion signal (ATR > 1.5x 20-period ATR MA).
# Session filter (08-20 UTC) to avoid low-liquidity hours. Discrete sizing 0.25.
# Target: 12-25 trades/year by combining 6h breakouts with 1d regime and 1w direction filters.

name = "6h_Donchian20_1dATRRegime_1wPivotDir_Session_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and its 20-period MA for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_regime = align_htf_to_ltf(prices, df_1d, atr_1d < atr_ma_20)  # True when ATR below MA (low vol)
    
    # Load 1w data ONCE before loop for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w Camarilla levels (S1, R1) from previous week OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla S1 and R1: C ± 1.1*(H-L)/6
    camarilla_s1_1w = close_1w - (1.1 * (high_1w - low_1w) / 6)
    camarilla_r1_1w = close_1w + (1.1 * (high_1w - low_1w) / 6)
    
    # Align weekly levels to 6h timeframe
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1_1w)
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr_regime = atr_regime[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_upper = highest_high[i]
        curr_lower = lowest_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 6h Donchian upper, low ATR regime, and above weekly S1
            if (curr_close > curr_upper and 
                curr_atr_regime and 
                curr_close > curr_s1):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h Donchian lower, low ATR regime, and below weekly R1
            elif (curr_close < curr_lower and 
                  curr_atr_regime and 
                  curr_close < curr_r1):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Donchian lower OR ATR expansion (regime change)
            if (curr_close < curr_lower or 
                not curr_atr_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Donchian upper OR ATR expansion (regime change)
            if (curr_close > curr_upper or 
                not curr_atr_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals