#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ChaikinVolatility_Expansion_Breakout_1dTrend"
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
    
    # 1d data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d Chaikin Volatility: EMA of (high-low) difference
    # Chaikin Volatility = EMA of (high - low) over period
    hl_diff = high_1d - low_1d
    chaikin_vol = pd.Series(hl_diff).ewm(span=10, adjust=False, min_periods=10).mean().values
    # Expansion signal: current Chaikin Volatility > 1.5 * 10-period SMA of Chaikin Volatility
    chaikin_sma10 = pd.Series(chaikin_vol).rolling(window=10, min_periods=10).mean().values
    chaikin_expansion = chaikin_vol > (chaikin_sma10 * 1.5)
    chaikin_expansion_aligned = align_htf_to_ltf(prices, df_1d, chaikin_expansion.astype(float))
    
    # 6h Donchian channel breakout (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(chaikin_expansion_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, 1d trend up (price > EMA50), volatility expanding
            long_cond = (close[i] > donchian_high[i] and 
                        close[i] > ema50_1d_aligned[i] and
                        chaikin_expansion_aligned[i] > 0.5)
            
            # Short: Price breaks below Donchian low, 1d trend down (price < EMA50), volatility expanding
            short_cond = (close[i] < donchian_low[i] and 
                         close[i] < ema50_1d_aligned[i] and
                         chaikin_expansion_aligned[i] > 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below Donchian low OR volatility contraction
            if close[i] < donchian_low[i] or chaikin_expansion_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above Donchian high OR volatility contraction
            if close[i] > donchian_high[i] or chaikin_expansion_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Chaikin Volatility expansion identifies periods of increasing volatility
# that often precede sustained moves. Combined with Donchian breakouts and 1d EMA50 trend
# filter, this captures institutional breakout moves in both bull and bear markets.
# The volatility filter avoids choppy markets while the trend filter ensures alignment
# with higher timeframe direction. 6h timeframe targets 15-35 trades/year to avoid fee drag.