#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ChopFilter_V2
Hypothesis: 4-hour Camarilla R1/S1 breakout with 1-day EMA34 trend filter and choppiness regime filter.
Targets 20-30 trades/year by requiring: 1) price breaks daily R1/S1 levels, 2) aligned with 1d EMA34 trend,
3) market in trending regime (Choppiness Index < 38.2). Uses discrete position sizing (0.25) to minimize fee churn.
Works in trending markets via breakout entries and exits at opposing Camarilla levels or trend change.
Avoids whipsaws in choppy markets via regime filter.
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
    
    # 1d data for Camarilla pivots, EMA34, and Choppiness Index (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (R1 = C + 1.1*(HL/4), S1 = C - 1.1*(HL/4))
    R1 = prev_close + 1.1 * prev_range * (1.0/4.0)
    S1 = prev_close - 1.1 * prev_range * (1.0/4.0)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d Choppiness Index (CHOP) - measures market choppiness/trendiness
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14))
    tr1 = np.abs(df_1d['high'] - df_1d['low'])
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14.sum() / np.log10(hh14 - ll14) / np.log10(14)) if (hh14 - ll14).any() > 0 else 50
    # Handle edge cases and compute properly
    chop_raw = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_sum = pd.Series(tr[i-13:i+1]).sum()
        hh = df_1d['high'].iloc[i-13:i+1].max()
        ll = df_1d['low'].iloc[i-13:i+1].min()
        if hh > ll:
            chop_raw[i] = 100 * np.log10(atr_sum / np.log10(hh - ll) / np.log10(14))
        else:
            chop_raw[i] = 50.0
    chop_values = chop_raw
    
    # Align 1d indicators to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d previous data (1) + 1d EMA34 (34) + 1d CHOP (14)
    start_idx = 34 + 14 + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Regime filter: trending market (CHOP < 38.2)
        trending_regime = chop_aligned[i] < 38.2
        
        if position == 0:
            # Look for entry signals with trend alignment and trending regime
            # Long breakout: price breaks above R1 with uptrend in trending regime
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and trending_regime
            # Short breakout: price breaks below S1 with downtrend in trending regime
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and trending_regime
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price breaks below S1 (mean reversion) or trend changes to downtrend or regime becomes choppy
            if curr_close < S1_aligned[i] or not uptrend or not trending_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above R1 (mean reversion) or trend changes to uptrend or regime becomes choppy
            if curr_close > R1_aligned[i] or not downtrend or not trending_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ChopFilter_V2"
timeframe = "4h"
leverage = 1.0