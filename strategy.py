#!/usr/bin/env python3
"""
4h_12h1d_price_channel_breakout_v1
Hypothesis: On 4h timeframe, trade breakouts of Donchian(20) channels aligned with 12h EMA trend and 1d low volatility regime. 
Go long when price breaks above Donchian upper band with 12h EMA up and 1d ATR percentile < 0.4 (low vol).
Go short when price breaks below Donchian lower band with 12h EMA down and 1d ATR percentile < 0.4.
Exit when price crosses back through Donchian midpoint or volatility increases (ATR percentile > 0.6).
Uses discrete position sizing (0.25) to minimize churn. Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h1d_price_channel_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 1d ATR for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR percentile rank (50-day lookback for stability)
    atr_percentile = pd.Series(atr_1d).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(atr_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility regime: ATR percentile below 40th percentile
        low_vol = atr_percentile_aligned[i] < 0.4
        
        # Trend direction from 12h EMA (using prior bar to avoid look-ahead)
        ema_up = ema_12h_aligned[i] > ema_12h_aligned[i-1] if i > 0 else False
        ema_down = ema_12h_aligned[i] < ema_12h_aligned[i-1] if i > 0 else False
        
        if position == 1:  # Long position
            # Exit: price crosses below midpoint or volatility increases
            if close[i] < donchian_mid[i] or atr_percentile_aligned[i] > 0.6:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above midpoint or volatility increases
            if close[i] > donchian_mid[i] or atr_percentile_aligned[i] > 0.6:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if low_vol:
                # Breakout above upper band with upward trend - go long
                if close[i] > highest_high[i] and ema_up:
                    position = 1
                    signals[i] = 0.25
                # Breakout below lower band with downward trend - go short
                elif close[i] < lowest_low[i] and ema_down:
                    position = -1
                    signals[i] = -0.25
    
    return signals