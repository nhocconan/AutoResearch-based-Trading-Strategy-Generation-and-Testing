#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR volatility filter
- Donchian(20) from 1w defines major support/resistance; breakout with ATR filter captures strong momentum
- 1w EMA50 defines the long-term trend: only long when price > EMA50, short when price < EMA50
- ATR(14) filter ensures breakouts occur during elevated volatility (ATR > 1.5x 50-period MA)
- Designed for 1d timeframe to capture major trend moves with low frequency (target: 7-25 trades/year)
- Uses 1w timeframe for structure to avoid noise and reduce false breakouts
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
    
    # Calculate 1w Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian channels: upper = max(high, 20), lower = min(low, 20)
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe (use previous week's levels for breakout)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ATR(14) for volatility filter (using 1d data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR MA for filter: ATR > 1.5x 50-period MA indicates elevated volatility
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14, 50)  # Donchian(20), EMA(50), ATR(14), ATR_MA(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND above 1w EMA50 AND elevated volatility
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                atr[i] > 1.5 * atr_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND below 1w EMA50 AND elevated volatility
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  atr[i] > 1.5 * atr_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level OR crosses 1w EMA50
            exit_signal = False
            
            if position == 1:
                # Exit long when price < Donchian lower OR < 1w EMA50
                if close[i] < donchian_lower_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > Donchian upper OR > 1w EMA50
                if close[i] > donchian_upper_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolatilityFilter"
timeframe = "1d"
leverage = 1.0