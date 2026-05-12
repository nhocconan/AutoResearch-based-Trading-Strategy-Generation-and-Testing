#!/usr/bin/env python3
name = "6h_Keltner_Channel_Squeeze_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # 1d trend filter: EMA34 (trend direction)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Keltner Channel on 6h: EMA20 center, ATR(10) * 2 for bands
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    upper = ema20 + 2 * atr
    lower = ema20 - 2 * atr
    
    # Bollinger Bands width for squeeze detection (20, 2)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    
    # Squeeze condition: BB width < Keltner width
    keltner_width = (upper - lower) / ema20
    squeeze = bb_width < keltner_width
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need 1d EMA34 and 6h EMA20
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long on squeeze breakout above upper band with 1d uptrend
            if (squeeze[i-1] if i > 0 else False) and close[i] > upper[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short on squeeze breakout below lower band with 1d downtrend
            elif (squeeze[i-1] if i > 0 else False) and close[i] < lower[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price closes below EMA20 or squeeze fires again
            if close[i] < ema20[i] or (squeeze[i] and close[i] < ema20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price closes above EMA20 or squeeze fires again
            if close[i] > ema20[i] or (squeeze[i] and close[i] > ema20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals