#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and ATR-based volatility filter.
- Long when price breaks above Donchian upper (20-bar high) AND close > 12h EMA50 AND ATR(14) < 1.5 * ATR(50) (low volatility regime)
- Short when price breaks below Donchian lower (20-bar low) AND close < 12h EMA50 AND ATR(14) < 1.5 * ATR(50)
- Exit on opposite Donchian breakout or trend reversal (close crosses 12h EMA50)
- Uses 4h primary timeframe with 12h HTF to target 75-200 total trades over 4 years (19-50/year)
- Donchian channels provide clear structure for breakouts in both ranging and trending markets
- 12h EMA50 ensures alignment with higher timeframe trend to avoid whipsaws
- ATR ratio filter ensures we only trade in low volatility regimes, reducing false breakouts during high volatility periods
- Designed for BTC/ETH with edge in capturing genuine breakouts while avoiding choppy markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Donchian channels (20-bar)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility filter: ATR(14) < 1.5 * ATR(50) (low volatility regime)
    vol_filter = atr_14 < (1.5 * atr_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper, trend up (close > EMA50), low volatility
            if close[i] > donchian_upper[i] and close[i] > ema_50_12h_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, trend down (close < EMA50), low volatility
            elif close[i] < donchian_lower[i] and close[i] < ema_50_12h_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower OR trend reversal (close < EMA50)
            if close[i] < donchian_lower[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper OR trend reversal (close > EMA50)
            if close[i] > donchian_upper[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_ATRVolFilter_v1"
timeframe = "4h"
leverage = 1.0