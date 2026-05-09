#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day structure. Uses daily ATR for volatility filter and daily EMA50 for trend.
# Trades breakouts of previous day's high/low with volatility-adjusted position sizing.
# Trend filter ensures trades align with higher timeframe direction.
# Volatility filter avoids low-volatility chop where breakouts fail.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_DailyBreakout_ATRFilter_EMA50"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Previous day's high/low for breakout levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Daily ATR for volatility filter (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate True Range for 1d
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h ATR for position sizing (volatility-adjusted)
    tr_12h = pd.Series(np.maximum(
        high - low,
        np.maximum(
            abs(high - np.roll(close, 1)),
            abs(low - np.roll(close, 1))
        )
    )).rolling(window=14, min_periods=14).mean().values
    atr_norm = tr_12h / (atr_14_1d_aligned + 1e-10)  # Normalize by daily ATR
    
    # Breakout conditions
    breakout_up = close > prev_high
    breakout_down = close < prev_low
    
    # Trend filter
    trend_up = close > ema_50_1d_aligned
    trend_down = close < ema_50_1d_aligned
    
    # Volatility filter: avoid low volatility (ATR ratio < 0.5) and extreme volatility (> 3.0)
    vol_filter = (atr_norm >= 0.5) & (atr_norm <= 3.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above prev day high + uptrend + adequate volatility
            if breakout_up[i] and trend_up[i] and vol_filter[i]:
                # Size inversely to volatility (but capped)
                size = 0.25 * min(1.0, 0.5 / max(atr_norm[i], 0.1))
                signals[i] = size
                position = 1
            # Short: breakout below prev day low + downtrend + adequate volatility
            elif breakout_down[i] and trend_down[i] and vol_filter[i]:
                size = 0.25 * min(1.0, 0.5 / max(atr_norm[i], 0.1))
                signals[i] = -size
                position = -1
        
        elif position == 1:
            # Exit long: price returns to prev day close or trend reversal
            if close[i] <= prev_close[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 * min(1.0, 0.5 / max(atr_norm[i], 0.1))
        
        elif position == -1:
            # Exit short: price returns to prev day close or trend reversal
            if close[i] >= prev_close[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 * min(1.0, 0.5 / max(atr_norm[i], 0.1))
    
    return signals