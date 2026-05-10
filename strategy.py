#!/usr/bin/env python3
"""
12h_Keltner_Squeeze_1dTrend_Volume
Hypothesis: Keltner Channel squeeze (BB inside KC) identifies low volatility periods.
Breakout from squeeze with volume confirmation and 1-day EMA34 trend filter captures
directional moves. Works in both bull/bear markets by trading breakouts in direction
of higher timeframe trend. Targets 15-35 trades/year on 12h timeframe.
"""

name = "12h_Keltner_Squeeze_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 12h data for Keltner Bands and signal generation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20, 2.0)
    # Middle = 20-period EMA
    # Width = 2.0 * ATR(10)
    # Upper = Middle + Width
    # Lower = Middle - Width
    
    # EMA20 for middle
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(10)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Bands
    kc_upper = ema20 + 2.0 * atr10
    kc_lower = ema20 - 2.0 * atr10
    
    # Bollinger Bands (20, 2.0) for squeeze detection
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    
    # Squeeze condition: BB inside KC (low volatility)
    squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 periods for BB/KC, 34 for 1d EMA
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema20[i]) or
            np.isnan(atr10[i]) or
            np.isnan(sma20[i]) or
            np.isnan(std20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1d EMA34
        uptrend_1d = close[i] > ema34_1d_aligned[i]
        downtrend_1d = close[i] < ema34_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period EMA
        vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
        volume_filter = volume[i] > vol_ema20[i] * 1.5
        
        if position == 0:
            # Look for breakout after squeeze
            if i > 0 and squeeze[i-1]:  # Was in squeeze on previous bar
                # Long breakout: price breaks above KC upper with volume and uptrend
                if close[i] > kc_upper[i] and volume_filter and uptrend_1d:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price breaks below KC lower with volume and downtrend
                elif close[i] < kc_lower[i] and volume_filter and downtrend_1d:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reaches KC middle or squeeze fires
            if close[i] <= ema20[i] or squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches KC middle or squeeze fires
            if close[i] >= ema20[i] or squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals