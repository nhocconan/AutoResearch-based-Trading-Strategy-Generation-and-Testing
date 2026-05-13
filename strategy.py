#!/usr/bin/env python3
"""
12h_Combined_Signal_Strategy
Hypothesis: Combines price action signals with momentum and volatility filters to capture trends while avoiding whipsaws.
Long when price breaks above Donchian high (20-period) with bullish momentum (MACD histogram > 0) and low volatility (ATR ratio < 1.2).
Short when price breaks below Donchian low with bearish momentum and low volatility.
Uses 1-day trend filter for higher timeframe bias.
Target: 20-40 trades/year per symbol.
"""

name = "12h_Combined_Signal_Strategy"
timeframe = "12h"
leverage = 1.0

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
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # MACD (12,26,9)
    ema_fast = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # ATR for volatility filter (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR ratio (current ATR / 50-period average ATR) for volatility regime
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma
    
    # 1-day trend filter: price vs 50-period EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Volatility filter: only trade when ATR ratio < 1.2 (low volatility)
        vol_filter = atr_ratio[i] < 1.2
        
        if position == 0:
            # LONG: price breaks above Donchian high, bullish momentum, uptrend, low volatility
            if close[i] > donch_high[i] and macd_hist[i] > 0 and uptrend_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low, bearish momentum, downtrend, low volatility
            elif close[i] < donch_low[i] and macd_hist[i] < 0 and downtrend_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian low or momentum turns bearish
            if close[i] < donch_low[i] or macd_hist[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian high or momentum turns bullish
            if close[i] > donch_high[i] or macd_hist[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals