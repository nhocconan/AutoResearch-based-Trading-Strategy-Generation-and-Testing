#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR-based stoploss.
- Primary timeframe: 1d for entries/exits.
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Entry: Long when price breaks above 20-day high AND 1w EMA50 bullish.
         Short when price breaks below 20-day low AND 1w EMA50 bearish.
- Exit: Opposite Donchian breakout or ATR-based stoploss (2.5 * ATR(14)).
- Signal size: 0.30 discrete to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
Donchian breakouts capture strong trends, EMA50 filter avoids counter-trend trades, ATR stop manages risk.
Works in both bull and bear markets by only trading in the direction of the 1w trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d Donchian channels (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0  # First bar has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    df_1w_close = df_1w['close'].values
    ema_50 = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need enough bars for Donchian, ATR, and 1w EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(atr[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            # Bullish: price breaks above 20-day high AND 1w EMA50 bullish (close > EMA50)
            if curr_high > period20_high[i] and curr_close > ema_50_aligned[i]:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Bearish: price breaks below 20-day low AND 1w EMA50 bearish (close < EMA50)
            elif curr_low < period20_low[i] and curr_close < ema_50_aligned[i]:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long exit: price breaks below 20-day low OR ATR stoploss hit
            if curr_low < period20_low[i] or curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above 20-day high OR ATR stoploss hit
            if curr_high > period20_high[i] or curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Donchian20_1wEMA50_ATRStop_v1"
timeframe = "1d"
leverage = 1.0