#!/usr/bin/env python3
name = "1d_KC_Breakout_Volume_Squeeze"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mpt_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # EMA50 on weekly for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Keltner Channel on daily
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    atr[:20] = np.nan
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # EMA20 for KC middle
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    # Bollinger Bands for squeeze detection (BB width < KC width)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = bb_upper - bb_lower
    kc_width = kc_upper - kc_lower
    squeeze = bb_width < kc_width
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > kc_upper[i-1]  # Previous bar's upper KC
        short_breakout = close[i] < kc_lower[i-1]  # Previous bar's lower KC
        
        if position == 0:
            # Long: bullish weekly trend, KC breakout, volume surge, squeeze condition
            if (close[i] > ema50_1w_aligned[i] and  # Price above weekly EMA50
                long_breakout and 
                volume[i] > 1.5 * vol_ma20[i] and
                squeeze[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish weekly trend, KC breakout, volume surge, squeeze condition
            elif (close[i] < ema50_1w_aligned[i] and  # Price below weekly EMA50
                  short_breakout and 
                  volume[i] > 1.5 * vol_ma20[i] and
                  squeeze[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KC middle OR trend changes
            if close[i] < ema20[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KC middle OR trend changes
            if close[i] > ema20[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals