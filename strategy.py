#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and 1w EMA200 trend filter.
- Primary timeframe: 12h for execution, HTF: 1d for ATR-based volatility filter, 1w for EMA200 trend direction.
- In strong uptrend (price > EMA200): long on breakout above 20-period Donchian high with volatility expansion (current ATR > 1.5 * 20-period ATR MA).
- In strong downtrend (price < EMA200): short on breakdown below 20-period Donchian low with volatility expansion.
- Exit: Opposite Donchian breakout or trend reversal (price crosses EMA200).
- Volume is NOT required; volatility expansion via ATR serves as confirmation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ATR (14-period) on 1d
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate EMA200 on 1w
    ema200 = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 12h
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    ema200_aligned = align_htf_to_ltf(prices, df_1w, ema200)
    
    # Calculate Donchian channels (20-period) on 12h
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR MA (20-period) on 12h for volatility filter
    # Need ATR on 12h first
    # True Range on 12h
    tr1_12h = pd.Series(high).diff().abs()
    tr2_12h = (pd.Series(high) - pd.Series(low.shift())).abs()
    tr3_12h = (pd.Series(low) - pd.Series(close.shift())).abs()
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h = tr_12h.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_12h = pd.Series(atr_12h).rolling(window=20, min_periods=20).mean().values
    
    # Volatility expansion: current ATR_12h > 1.5 * 20-period ATR MA
    vol_expansion = atr_12h > (1.5 * atr_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 20, 20)  # Need enough for EMA200, Donchian, ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_aligned[i]) or np.isnan(ema200_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_expansion[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema200_val = ema200_aligned[i]
        vol_exp = vol_expansion[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        
        if position == 0:
            # Check for entry signals
            if vol_exp:
                # Strong uptrend: price above EMA200
                if curr_close > ema200_val:
                    # Long on breakout above Donchian high
                    if curr_high > dh:
                        signals[i] = 0.25
                        position = 1
                # Strong downtrend: price below EMA200
                elif curr_close < ema200_val:
                    # Short on breakdown below Donchian low
                    if curr_low < dl:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR trend reverses (price < EMA200)
            if curr_low < dl or curr_close < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR trend reverses (price > EMA200)
            if curr_high > dh or curr_close > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATRFilter_1wEMA200Trend_v1"
timeframe = "12h"
leverage = 1.0