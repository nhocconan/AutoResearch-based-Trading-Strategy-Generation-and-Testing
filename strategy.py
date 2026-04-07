#!/usr/bin/env python3
"""
1h_4h1d_trend_volatility_breakout_v1
Hypothesis: In volatile crypto markets, strong momentum bursts often precede sustained moves. Use 4h trend (EMA21) for direction, 1d volatility regime (ATR ratio) to filter chop, and 1h Donchian breakout for precise entry. This captures momentum bursts while avoiding sideways chop. Target: 15-37 trades/year (60-150 over 4 years) to minimize fee drag. Works in bull/bear via volatility regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_trend_volatility_breakout_v1"
timeframe = "1h"
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
    
    # 1. 4h Trend Filter (EMA21) - gets direction from higher timeframe
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 2. 1d Volatility Regime Filter (ATR ratio) - avoids choppy markets
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])  # First bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # First bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Current ATR (14-period on 1d)
    atr_ratio = atr_1d / pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 3. 1h Donchian Breakout (20-period) - entry timing
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: only trade when volatility is elevated (trending market)
        vol_regime = atr_ratio_aligned[i] > 1.1
        
        if position == 1:  # Long position
            # Exit: trend reversal OR volatility collapse
            if close[i] <= ema_4h_aligned[i] or atr_ratio_aligned[i] < 0.9:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: trend reversal OR volatility collapse
            if close[i] >= ema_4h_aligned[i] or atr_ratio_aligned[i] < 0.9:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if vol_regime:
                # Long breakout: price breaks above Donchian high with uptrend
                if close[i] > donchian_high[i] and close[i] > ema_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short breakout: price breaks below Donchian low with downtrend
                elif close[i] < donchian_low[i] and close[i] < ema_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals