# 4H_KAMA_CROSSOVER_VOLUME_ADX
# Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely,
# in ranging markets it stays flat. Crossovers of fast/slow KAMA capture trend changes
# with reduced whipsaw. Volume confirmation ensures institutional participation.
# ADX > 25 filters for trending markets only. Works in both bull and bear regimes.
# Target: 25-40 trades/year to minimize fee drag while capturing significant moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX for regime filter (slower but more reliable)
    adx_len = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # KAMA parameters
    fast_len = 2
    slow_len = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    abs_change = np.sum(np.abs(np.diff(close, n=1))[:, None] * np.ones((1, 10)), axis=1)
    abs_change = np.concatenate([np.full(10, np.nan), abs_change])
    er = np.where(abs_change != 0, change / abs_change, 0)
    
    # Smoothing constants
    fast_sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
    slow_sc = (2/(slow_len+1)) ** 2
    
    # KAMA calculation
    kama_fast = np.zeros_like(close)
    kama_slow = np.zeros_like(close)
    kama_fast[0] = close[0]
    kama_slow[0] = close[0]
    
    for i in range(1, n):
        kama_fast[i] = kama_fast[i-1] + fast_sc[i] * (close[i] - kama_fast[i-1])
        kama_slow[i] = kama_slow[i-1] + slow_sc[i] * (close[i] - kama_slow[i-1])
    
    # Volume average (50 periods for stability)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, 50)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_fast[i]) or 
            np.isnan(kama_slow[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: fast KAMA crosses above slow KAMA + volume + trend
            if (kama_fast[i] > kama_slow[i] and 
                kama_fast[i-1] <= kama_slow[i-1] and
                volume_confirmed and 
                trending):
                position = 1
                signals[i] = position_size
            # Enter short: fast KAMA crosses below slow KAMA + volume + trend
            elif (kama_fast[i] < kama_slow[i] and 
                  kama_fast[i-1] >= kama_slow[i-1] and
                  volume_confirmed and 
                  trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: fast KAMA crosses below slow KAMA
            if kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: fast KAMA crosses above slow KAMA
            if kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4H_KAMA_Crossover_Volume_ADX"
timeframe = "4h"
leverage = 1.0