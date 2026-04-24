#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and ATR volatility filter.
- Long when price breaks above 20-period Donchian high AND 1w close > 1w EMA50 (bullish regime)
- Short when price breaks below 20-period Donchian low AND 1w close < 1w EMA50 (bearish regime)
- ATR filter: only trade when ATR(14) > 0.5 * 20-period ATR mean (avoid low volatility chop)
- Exit on opposite Donchian level or when trend filter reverses
- Uses 12h primary with 1w HTF to target 50-150 total trades over 4 years (12-37/year)
- Donchian provides clear breakout levels; EMA50 filters regime; ATR filter avoids whipsaws
- Designed to work in both bull (breakouts with trend) and bear (breakouts against trend) markets
- Signal size: 0.30 discrete levels to balance return and fee drag
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_1w_aligned
    bearish_regime = close < ema_50_1w_aligned
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR filter: avoid low volatility regimes
    atr_mean = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_filter = atr > (0.5 * atr_mean)
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14, 20) + 1  # Need EMA50, Donchian20, ATR14, ATR mean
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high AND bullish regime AND sufficient volatility
            if close[i] > donchian_high[i] and bullish_regime[i] and atr_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: break below Donchian low AND bearish regime AND sufficient volatility
            elif close[i] < donchian_low[i] and bearish_regime[i] and atr_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: break below Donchian low OR bearish regime
            if close[i] < donchian_low[i] or not bullish_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: break above Donchian high OR bullish regime
            if close[i] > donchian_high[i] or not bearish_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_1wEMA50_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0