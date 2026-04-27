#!/usr/bin/env python3
"""
6h_Engulfing_BullBear_Engulf_Trend_1d
Hypothesis: On 6h candles, detect bullish/bearish engulfing patterns (strong reversal signal) and trade in the direction of the daily EMA50 trend. Engulfing patterns indicate momentum shifts, and filtering by daily trend avoids counter-trend trades. Works in bull markets (buy the dip in uptrend) and bear markets (sell the rally in downtrend). Targets ~20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Detect engulfing patterns on 6h candles
    # Bullish engulfing: current green candle fully engulfs previous red candle
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close)  # Placeholder, will fix below
    
    # Actually compute engulfing properly
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close.shift(1) < open_price.shift(1)) & \
                     (close > open_price.shift(1)) & \
                     (open_price < close.shift(1))
    
    # Bearish engulfing: current red candle fully engulfs previous green candle
    bearish_engulf = (close < open_price) & (open_price > close) & \
                     (close.shift(1) > open_price.shift(1)) & \
                     (close < open_price.shift(1)) & \
                     (open_price > close.shift(1))
    
    # Handle NaN from shift
    bullish_engulf = np.where(np.isnan(bullish_engulf), False, bullish_engulf)
    bearish_engulf = np.where(np.isnan(bearish_engulf), False, bearish_engulf)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and engulfing (need previous candle)
    start_idx = max(50, 1)
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_1d_aligned[i]
        bull_eng = bullish_engulf[i]
        bear_eng = bearish_engulf[i]
        
        if position == 0:
            # Long: bullish engulfing + uptrend (close > EMA50)
            if bull_eng and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: bearish engulfing + downtrend (close < EMA50)
            elif bear_eng and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish engulfing or trend turns down
            if bear_eng or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish engulfing or trend turns up
            if bull_eng or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Engulfing_BullBear_Engulf_Trend_1d"
timeframe = "6h"
leverage = 1.0