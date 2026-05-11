#!/usr/bin/env python3
"""
6h_Engulfing_Reversal_TrendFilter
Hypothesis: On 6h timeframe, bullish/bearish engulfing candles at key 12h support/resistance levels
with 12h EMA50 trend filter provide high-probability reversals in both bull and bear markets.
Engulfing patterns signal strong momentum shifts; 12h EMA50 filters counter-trend noise.
Target: 20-40 trades/year (80-160 total over 4 years) to avoid fee drag.
"""

name = "6h_Engulfing_Reversal_TrendFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend and key levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h EMA50 trend ---
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # --- 12h recent swing high/low for support/resistance ---
    # Use 20-period lookback for swing points
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate rolling max/min for swing levels
    swing_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    swing_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # --- 6h engulfing detection ---
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulf = (
        (close > open_price) &  # current bullish
        (open_price < close) &  # redundant but clear
        (open_price <= close) & 
        (close[-1:] > open_price[:-1]) &  # current close > prev open
        (open_price < close[:-1]) &       # current open < prev close
        (close[:-1] < open_price[:-1])    # prev bearish
    )
    # Shift to align with current bar (we need previous bar data)
    bullish_engulf = np.roll(bullish_engulf, 1)
    bullish_engulf[0] = False
    
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulf = (
        (close < open_price) &  # current bearish
        (open_price >= close) & 
        (close[-1:] < open_price[:-1]) &  # current close < prev open
        (open_price > close[:-1]) &       # current open > prev close
        (close[:-1] > open_price[:-1])    # prev bullish
    )
    bearish_engulf = np.roll(bearish_engulf, 1)
    bearish_engulf[0] = False
    
    # Align 12h indicators to 6h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    swing_high_12h_aligned = align_htf_to_ltf(prices, df_12h, swing_high_12h)
    swing_low_12h_aligned = align_htf_to_ltf(prices, df_12h, swing_low_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(12h EMA50, 20-period swing)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_12h_aligned[i]) or
            np.isnan(swing_high_12h_aligned[i]) or
            np.isnan(swing_low_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: bullish engulfing at or above 12h swing low, with 12h uptrend
            if (bullish_engulf[i] and 
                low[i] <= swing_low_12h_aligned[i] * 1.005 and  # near support
                ema_12h_aligned[i] > ema_12h_aligned[i-1]):    # 12h uptrend
                signals[i] = 0.25
                position = 1
            # Short setup: bearish engulfing at or below 12h swing high, with 12h downtrend
            elif (bearish_engulf[i] and 
                  high[i] >= swing_high_12h_aligned[i] * 0.995 and  # near resistance
                  ema_12h_aligned[i] < ema_12h_aligned[i-1]):     # 12h downtrend
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: bearish engulfing or 12h trend turns down
                if bearish_engulf[i] or ema_12h_aligned[i] < ema_12h_aligned[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish engulfing or 12h trend turns up
                if bullish_engulf[i] or ema_12h_aligned[i] > ema_12h_aligned[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals