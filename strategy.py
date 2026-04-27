#!/usr/bin/env python3
"""
4h_3BarReversal_LiquidityCave_Momentum
Hypothesis: Three-bar reversal patterns at liquidity caves (equal highs/lows) with momentum confirmation capture reversals in both trending and ranging markets.
Works in bull/bear by requiring momentum confirmation and avoiding counter-trend entries. Targets 20-30 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate ATR(14) for stop loss
    high_low = high - low
    high_close_prev = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    low_close_prev = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema20_1d_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema20_1d_aligned[i]
        
        # Three-bar reversal pattern
        # Bullish: three consecutive higher lows
        bullish_3br = (low[i] > low[i-1]) and (low[i-1] > low[i-2])
        # Bearish: three consecutive lower highs
        bearish_3br = (high[i] < high[i-1]) and (high[i-1] < high[i-2])
        
        # Liquidity cave: equal highs/lows within 0.1% tolerance
        equal_high = abs(high[i] - high[i-1]) / high[i] < 0.001
        equal_low = abs(low[i] - low[i-1]) / low[i] < 0.001
        liquidity_cave = equal_high or equal_low
        
        # Momentum confirmation: price > VWAP for long, price < VWAP for short
        vwap_num = (high + low + close) * volume
        vwap_den = volume
        vwap = np.nancumsum(vwap_num) / np.nancumsum(vwap_den)
        price_above_vwap = close[i] > vwap[i]
        price_below_vwap = close[i] < vwap[i]
        
        if position == 0:
            # Long: bullish 3-bar reversal at liquidity cave with momentum and uptrend
            if bullish_3br and liquidity_cave and price_above_vwap and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: bearish 3-bar reversal at liquidity cave with momentum and downtrend
            elif bearish_3br and liquidity_cave and price_below_vwap and close[i] < ema_trend:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: bearish reversal or stop loss
            if bearish_3br or close[i] < (high[i-1] - 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish reversal or stop loss
            if bullish_3br or close[i] > (low[i-1] + 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_3BarReversal_LiquidityCave_Momentum"
timeframe = "4h"
leverage = 1.0