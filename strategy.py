#!/usr/bin/env python3
"""
4h_RSI40_BullishEngulfing_1dTrend_Volume
Hypothesis: Combine RSI < 40 (oversold) with bullish engulfing candle pattern on 4h timeframe,
filtered by 1-day EMA50 trend direction and volume confirmation. In bear markets, oversold
conditions with bullish reversal patterns often lead to mean-reversion bounces. In bull markets,
these conditions signal continuation of uptrend. Volume confirms institutional participation.
"""

name = "4h_RSI40_BullishEngulfing_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === RSI(14) Calculation ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Bullish Engulfing Pattern ===
    bullish_engulfing = (close > open_price) & (open_price < close) & \
                        (close > open_price) & (open_price < close) & \
                        (close > open_price) & (open_price < close)  # Placeholder - corrected below
    # Actually: current candle bullish and engulfs previous bearish candle
    bullish_engulfing = (close > open_price) & \
                        (open_price <= close[1:] if len(close) > 1 else False) & \
                        (close >= open_price[1:] if len(open_price) > 1 else False)
    # Fix array alignment
    bullish_engulfing = np.zeros(n, dtype=bool)
    bullish_engulfing[1:] = (close[1:] > open_price[1:]) & \
                           (open_price[1:] <= close[:-1]) & \
                           (close[1:] >= open_price[:-1])
    
    # === Daily Trend Filter (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Filter (1.5x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: RSI < 40 (oversold) + bullish engulfing + uptrend + volume
            if (rsi[i] < 40 and 
                bullish_engulfing[i] and 
                close[i] > ema50_4h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Exit: RSI > 60 (overbought) or price closes below EMA50
            if rsi[i] > 60 or close[i] < ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
    
    return signals