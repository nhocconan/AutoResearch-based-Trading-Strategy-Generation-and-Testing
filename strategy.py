#!/usr/bin/env python3
# 4h_Keltner_Channel_Squeeze_Breakout_Trend_Filter
# Hypothesis: Keltner Channel squeeze breakout with trend filter and volume confirmation.
# The strategy identifies low volatility periods (squeeze) where price breaks out of the Keltner Channel
# with volume confirmation and trend alignment. Works in both bull and bear markets by capturing
# volatility expansion after contraction periods.
# Uses Keltner Channel (20, 1.5) with ATR(10) for bands, EMA(50) for trend filter.

name = "4h_Keltner_Channel_Squeeze_Breakout_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(10) for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate EMA(20) for Keltner Channel basis
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands
    kc_upper = ema_20 + (atr * 1.5)
    kc_lower = ema_20 - (atr * 1.5)
    
    # Bollinger Bands for squeeze detection (20, 2.0)
    bb_middle = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (bb_std * 2.0)
    bb_lower = bb_middle - (bb_std * 2.0)
    
    # Squeeze condition: BB inside KC (low volatility)
    squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA(50) daily (50), ATR(10) (10), EMA(20) (20), BB (20), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Breakout conditions
        breakout_up = close[i] > kc_upper[i]
        breakout_down = close[i] < kc_lower[i]
        
        if position == 0:
            # Long entry: squeeze breakout up + uptrend + volume
            if squeeze[i-1] and breakout_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: squeeze breakout down + downtrend + volume
            elif squeeze[i-1] and breakout_down and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA(20) or trend changes
            if close[i] < ema_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above EMA(20) or trend changes
            if close[i] > ema_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals