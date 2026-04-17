#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR(14) volatility filter.
Long when price breaks above upper Donchian channel AND price > 1w EMA50 AND ATR(14) > ATR(50) (high volatility regime).
Short when price breaks below lower Donchian channel AND price < 1w EMA50 AND ATR(14) > ATR(50).
Exit when price crosses the 10-period EMA of closes (mean reversion) or volatility collapses (ATR(14) < 0.5 * ATR(50)).
Uses 1d for price action and Donchian channels, 1w for EMA50 trend filter and ATR regime filter.
Target: 30-100 total trades over 4 years (7-25/year). Donchian breakouts capture strong momentum, 
weekly EMA50 ensures trend alignment, ATR filter avoids low-volatility false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime filter
    high_1d_s = pd.Series(high_1d)
    low_1d_s = pd.Series(low_1d)
    close_1d_s = pd.Series(close_1d)
    
    tr1 = high_1d_s - low_1d_s
    tr2 = abs(high_1d_s - close_1d_s.shift(1))
    tr3 = abs(low_1d_s - close_1d_s.shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50_1d = tr_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d Donchian(20) channels
    upper_20 = high_1d_s.rolling(window=20, min_periods=20).max().values
    lower_20 = low_1d_s.rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    close_1w_s = pd.Series(close_1w)
    ema50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d EMA10 for exit signal
    close_s = pd.Series(close)
    ema10 = close_s.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Align 1d indicators to 1d timeframe (no alignment needed as we're already on 1d)
    # But we need to align for consistency with MTF approach
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    ema10_aligned = align_htf_to_ltf(prices, df_1d, ema10)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        ema50 = ema50_1w_aligned[i]
        atr14 = atr14_1d_aligned[i]
        atr50 = atr50_1d_aligned[i]
        ema10_val = ema10_aligned[i]
        
        # Volatility regime: high volatility when ATR(14) > ATR(50)
        high_vol_regime = atr14 > atr50
        low_vol_regime = atr14 < 0.5 * atr50
        
        if position == 0:
            # Long: price breaks above upper Donchian AND weekly uptrend AND high volatility
            if price > upper and price > ema50 and high_vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND weekly downtrend AND high volatility
            elif price < lower and price < ema50 and high_vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA10 OR volatility collapses
            if price < ema10_val or low_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA10 OR volatility collapses
            if price > ema10_val or low_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA50_ATRRegime"
timeframe = "1d"
leverage = 1.0