#!/usr/bin/env python3
name = "6h_Liquidity_Trap_Reversal"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data ONCE for liquidity trap detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily ATR for volatility filter
    atr_14_1d = pd.Series(df_1d['close']).rolling(window=14, min_periods=14).apply(
        lambda x: np.sqrt(np.mean((np.diff(x) ** 2))), raw=True
    ).values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Liquidity trap detection: price tests recent high/low but fails to break
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, lookback)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Avoid trading when volatility is too low
        if atr_14_1d_aligned[i] < np.mean(atr_14_1d_aligned[max(0, i-50):i]) * 0.5:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Liquidity trap long: price tests recent low but closes above it with volume
        trap_long = (low[i] <= lowest_low[i] * 1.001) and (close[i] > lowest_low[i]) and (volume[i] > vol_ma_20[i] * 1.5)
        
        # Liquidity trap short: price tests recent high but closes below it with volume
        trap_short = (high[i] >= highest_high[i] * 0.999) and (close[i] < highest_high[i]) and (volume[i] > vol_ma_20[i] * 1.5)
        
        if position == 0:
            # Long trap: expect reversal up
            if trap_long and ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short trap: expect reversal down
            elif trap_short and ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks above recent high or trap fails
            if close[i] > highest_high[i] or not trap_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks below recent low or trap fails
            if close[i] < lowest_low[i] or not trap_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Liquidity Trap Reversal
# - Identifies when price tests recent swing highs/lows but fails to break (liquidity grab)
# - Uses daily ATR for volatility filter to avoid choppy markets
# - Weekly EMA20 trend filter ensures alignment with higher timeframe trend
# - Entry: price tests liquidity level and reverses with volume confirmation
# - Exit: price breaks through the liquidity level or trap condition fails
# - Works in both bull (traps at support in uptrend) and bear (traps at resistance in downtrend)
# - Volume confirmation (1.5x average) reduces false signals
# - Position size 0.25 targets ~50-150 trades over 4 years (12-37/year)
# - Novel approach: focuses on failed breakouts rather than breakouts themselves
# - Effective in ranging markets where liquidity hunts are common
# - Weekly trend filter prevents counter-trend traps in strong trends