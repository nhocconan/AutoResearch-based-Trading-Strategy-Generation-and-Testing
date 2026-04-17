#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1-day ATR-based volatility breakout with 1-week EMA trend filter and volume confirmation.
- Calculate ATR(14) on 1-day timeframe for volatility measurement
- Enter long when price breaks above previous 1-day high + 0.5 * ATR(14) with volume > 1.3x 20-period volume MA and price above 1-week EMA50
- Enter short when price breaks below previous 1-day low - 0.5 * ATR(14) with volume > 1.3x 20-period volume MA and price below 1-week EMA50
- Exit when price returns to the previous 1-day close level
- Fixed position size 0.25 to manage drawdown
- Uses 1-week trend filter to avoid counter-trend trades
- Designed for 12h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
"""

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
    volume = prices['volume'].values
    
    # Get 1-day data for volatility breakout calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day ATR(14) for volatility measurement
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # First value has no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[13] = np.mean(tr[:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volatility breakout levels from previous day's OHLC + ATR
    # Long breakout: previous day high + 0.5 * ATR(14)
    # Short breakout: previous day low - 0.5 * ATR(14)
    # Exit level: previous day close
    high_breakout = high_1d + 0.5 * atr_14_1d
    low_breakout = low_1d - 0.5 * atr_14_1d
    exit_level = close_1d
    
    # Align levels to 12h timeframe (use previous day's levels)
    high_breakout_aligned = align_htf_to_ltf(prices, df_1d, high_breakout)
    low_breakout_aligned = align_htf_to_ltf(prices, df_1d, low_breakout)
    exit_level_aligned = align_htf_to_ltf(prices, df_1d, exit_level)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(high_breakout_aligned[i]) or np.isnan(low_breakout_aligned[i]) or 
            np.isnan(exit_level_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        high_break = high_breakout_aligned[i]
        low_break = low_breakout_aligned[i]
        exit_level_val = exit_level_aligned[i]
        ema_val = ema_50_1w_aligned[i]
        
        if position == 0:
            # Look for volatility breakouts with volume confirmation and trend filter
            # Long: price breaks above previous day high + 0.5*ATR + volume spike + price above 1w EMA50
            if price > high_break and vol > 1.3 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below previous day low - 0.5*ATR + volume spike + price below 1w EMA50
            elif price < low_break and vol > 1.3 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to previous day's close level
            if price < exit_level_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to previous day's close level
            if price > exit_level_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VolatilityBreakout_ATR_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0