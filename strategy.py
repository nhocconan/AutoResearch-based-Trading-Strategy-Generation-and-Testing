#!/usr/bin/env python3
"""
6h_MarketFacets_v1
Hypothesis: Combines market facets - trend (ADX), momentum (ROC), and volatility (ATR) on 6h timeframe with 1d trend filter.
Long when: ADX > 25 (trending), ROC > 0 (positive momentum), price > ATR-based support, and above 1d EMA50.
Short when: ADX > 25, ROC < 0, price < ATR-based resistance, and below 1d EMA50.
Uses volatility-adjusted entry/exit to whipsaw in ranging markets while capturing trends.
Designed for low trade frequency by requiring ADX trending condition plus momentum confirmation.
Works in bull markets via momentum longs and bear markets via momentum shorts.
"""

name = "6h_MarketFacets_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- ADX Calculation (14-period) ---
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    plus_di = 100 * (np.cumsum(plus_dm) / np.cumsum(atr))
    minus_di = 100 * (np.cumsum(minus_dm) / np.cumsum(atr))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = np.zeros(n)
    adx[0] = dx[0]
    for i in range(1, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # --- Rate of Change (10-period) ---
    roc = np.zeros(n)
    roc[:10] = np.nan
    for i in range(10, n):
        roc[i] = ((close[i] - close[i-10]) / close[i-10]) * 100
    
    # --- ATR-based Support/Resistance (2*ATR from recent swing) ---
    # Support: lowest low minus 2*ATR over last 20 periods
    # Resistance: highest high plus 2*ATR over last 20 periods
    lowest_low = np.zeros(n)
    highest_high = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i-19)
        lowest_low[i] = np.min(low[start_idx:i+1])
        highest_high[i] = np.max(high[start_idx:i+1])
    
    support = lowest_low - 2 * atr
    resistance = highest_high + 2 * atr
    
    # --- 1d Trend Filter (EMA50) ---
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(roc[i]) or np.isnan(support[i]) or 
            np.isnan(resistance[i]) or np.isnan(ema_50_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: ADX trending, positive momentum, above support, above 1d EMA50
            if (adx[i] > 25 and roc[i] > 0 and 
                close[i] > support[i] and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: ADX trending, negative momentum, below resistance, below 1d EMA50
            elif (adx[i] > 25 and roc[i] < 0 and 
                  close[i] < resistance[i] and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: loss of trend or momentum reversal
            if position == 1:
                # Exit long: ADX weak or momentum turns negative
                if adx[i] < 20 or roc[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: ADX weak or momentum turns positive
                if adx[i] < 20 or roc[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals