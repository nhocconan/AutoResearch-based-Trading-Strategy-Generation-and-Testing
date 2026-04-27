#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_RegimeFilter
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA34 trend filter and choppiness regime.
Works in bull/bear: In trending markets (price > weekly EMA34 for longs, < for shorts),
breakouts at R1/S1 with low choppiness (trending regime) capture momentum. Uses discrete
position sizing (0.25) to limit drawdown and reduce fee churn. Targets 20-80 trades over 4 years
on 1d timeframe. Weekly HTF ensures structural alignment, daily timeframe avoids overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter and choppiness
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA34 for trend
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Weekly choppiness: CHOP(14) = 100 * log10(sum(ATR(14)) / log10(highest-high - lowest-low) / 14)
    # Simplified: use rolling ATR ratio as proxy for chop
    tr = np.maximum(df_1w['high'].values - df_1w['low'].values,
                    np.maximum(abs(df_1w['high'].values - df_1w['close'].shift(1).values),
                               abs(df_1w['low'].values - df_1w['close'].shift(1).values)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh_14 = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14) / np.log10(hh_14 - ll_14 + 1e-10)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels from previous daily bar (completed)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align Camarilla levels to daily (no shift needed as 1d->1d)
    r1_aligned = r1  # Already aligned to daily close
    s1_aligned = s1
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position
    
    # Warmup: need weekly EMA34, chop, and daily shift
    start_idx = max(34, 14, 1) + 1  # +1 for shift(1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_34_aligned[i]
        chop_val = chop_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # Regime filter: only trade in trending regime (CHOP < 38.2)
        trending_regime = chop_val < 38.2
        
        if position == 0:
            # Look for entry: Camarilla breakout with weekly trend alignment and trending regime
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            trending_regime)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             trending_regime)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price re-enters Camarilla range (below S1) OR loses weekly trend OR chop becomes high
            if close_val < s1_val or close_val < ema_val or chop_val >= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price re-enters Camarilla range (above R1) OR loses weekly trend OR chop becomes high
            if close_val > r1_val or close_val > ema_val or chop_val >= 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_RegimeFilter"
timeframe = "1d"
leverage = 1.0