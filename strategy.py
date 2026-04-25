#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Volume_RegimeFilter
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume confirmation, and choppiness regime filter.
Designed for 12-37 trades/year on BTC/ETH in both bull and bear markets by combining price structure (Camarilla),
trend filter (1d EMA), momentum (volume spike), and regime filter (choppiness < 61.8 for trending markets).
Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and EMA34 trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (inner levels for breakouts)
    R1 = prev_close + 1.1 * prev_range * 0.125  # R1 = C + 1.1*(HL/8)
    S1 = prev_close - 1.1 * prev_range * 0.125  # S1 = C - 1.1*(HL/8)
    
    # Align 1d R1/S1 levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness regime filter: CHOP < 61.8 = trending (use trend following)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum_tr / log10(hh - ll)) / log10(14)
    # Avoid division by zero and log of zero/negative
    hl_range = hh - ll
    chop = np.zeros(n)
    for i in range(n):
        if hl_range[i] <= 0 or sum_tr[i] <= 0:
            chop[i] = 50.0  # neutral
        else:
            chop[i] = 100.0 * np.log10(sum_tr[i]) / np.log10(14) / np.log10(hl_range[i])
    
    # Trending regime: CHOP < 61.8
    trending_regime = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA34 (34) and CHOP (14+14-1=27) -> max 34
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume spike, trend alignment, and trending regime
            # Long breakout: price breaks above R1 with uptrend, volume spike, and trending regime
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_spike[i] and trending_regime[i]
            # Short breakout: price breaks below S1 with downtrend, volume spike, and trending regime
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_spike[i] and trending_regime[i]
            
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
            # Long position: exit conditions
            # Stoploss: 2.0 * ATR below entry
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks below S1 (mean reversion) or trend changes or regime shifts to choppy
            elif curr_close < S1_aligned[i] or not uptrend or not trending_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: 2.0 * ATR above entry
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks above R1 (mean reversion) or trend changes or regime shifts to choppy
            elif curr_close > R1_aligned[i] or not downtrend or not trending_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume_RegimeFilter"
timeframe = "12h"
leverage = 1.0