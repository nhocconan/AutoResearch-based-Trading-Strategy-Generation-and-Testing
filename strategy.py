#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_Regime
Hypothesis: On 12h timeframe, TRIX (triple exponential average) crossover signals with volume confirmation (>1.5x 20-period MA) and choppiness regime filter (CHOP > 61.8 = range) captures mean-reversion opportunities in choppy markets and trend continuations in trending markets. The strategy uses discrete sizing (±0.30) and ATR-based trailing stop (2.0x) to minimize fee drag and works in both bull/bear markets with BTC/ETH edge.
"""

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
    
    # Load 1d data ONCE before loop for TRIX and choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate TRIX on 1d: triple EMA of ROC
    close_1d = df_1d['close'].values
    # ROC(1) = (close - close.shift(1)) / close.shift(1)
    roc = np.diff(close_1d) / close_1d[:-1]
    roc = np.concatenate([[np.nan], roc])  # align with original length
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3 * 100  # scale for readability
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Calculate Choppiness Index on 1d: CHOP > 61.8 = range, CHOP < 38.2 = trend
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(high_1d) - pd.Series(close_1d).shift()
    tr3 = pd.Series(low_1d) - pd.Series(close_1d).shift()
    tr_1d = pd.concat([tr1.abs(), tr2.abs(), tr3.abs()], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean().values
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (max_high - min_low)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h ATR(20) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.ewm(span=20, adjust=False, min_periods=20).mean()
    atr_12h_values = atr_12h.values
    
    # Volume spike filter: volume > 1.5 * 20-period MA on 12h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of TRIX (15*3=45), ATR (20), volume MA (20), chop (14) + time for 1d alignment
    start_idx = max(45, 20, 20, 14) + 16  # +16 to ensure 1d bar completion (12h -> 1d: 2 bars per day)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        trix_val = trix_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_12h_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(trix_val) or np.isnan(chop_val) or np.isnan(atr_val) or 
            np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # TRIX signal: bullish when TRIX > 0 and rising, bearish when TRIX < 0 and falling
        # Use previous TRIX to determine slope
        if i > start_idx:
            trix_prev = trix_aligned[i-1]
            trix_rising = trix_val > trix_prev
            trix_falling = trix_val < trix_prev
        else:
            trix_rising = False
            trix_falling = False
        
        # Regime filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (trend follow)
        is_range = chop_val > 61.8
        is_trend = chop_val < 38.2
        
        # Entry logic: in range, mean reversion off TRIX extremes; in trend, trend continuation
        long_entry = False
        short_entry = False
        
        if is_range:
            # In range: mean reversion - long when TRIX oversold and rising, short when overbought and falling
            long_entry = (trix_val < -0.5) and trix_rising and vol_spike
            short_entry = (trix_val > 0.5) and trix_falling and vol_spike
        elif is_trend:
            # In trend: trend continuation - long when TRIX > 0 and rising, short when TRIX < 0 and falling
            long_entry = (trix_val > 0) and trix_rising and vol_spike
            short_entry = (trix_val < 0) and trix_falling and vol_spike
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.0 * ATR
            stop_price = highest_since_long - 2.0 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.0 * ATR
            stop_price = lowest_since_short + 2.0 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_TRIX_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0