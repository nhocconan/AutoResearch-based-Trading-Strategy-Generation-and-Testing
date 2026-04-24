#!/usr/bin/env python3
"""
Hypothesis: 1h EMA crossover with 4h/1d regime filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA50 trend filter and 1d for ATR volume spike filter.
- Session filter: 08-20 UTC to reduce noise trades.
- Entry: Long when 1h EMA12 crosses above EMA26 AND price > 4h EMA50 AND 1d ATR ratio > 1.8 AND session active.
         Short when 1h EMA12 crosses below EMA26 AND price < 4h EMA50 AND 1d ATR ratio > 1.8 AND session active.
- Exit: Opposite EMA crossover OR price crosses 4h EMA50 in opposite direction.
- Signal size: 0.20 discrete to minimize fee drag.
- ATR ratio (current ATR/20-period ATR) > 1.8 confirms volatility expansion to avoid false signals.
- 4h EMA50 provides trend filter to avoid counter-trend trades.
- Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on EMA crossover frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1h EMAs for crossover
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    
    # Calculate 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    ema50_4h = ema(df_4h['close'].values, 50)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_ema12 = ema12[i]
        curr_ema26 = ema26[i]
        prev_ema12 = ema12[i-1]
        prev_ema26 = ema26[i-1]
        
        # Exit conditions: opposite EMA crossover OR price crosses 4h EMA50 in opposite direction
        if position != 0:
            # Exit long: EMA12 crosses below EMA26 OR price falls below 4h EMA50
            if position == 1:
                if curr_ema12 < curr_ema26 and prev_ema12 >= prev_ema26:  # bearish crossover
                    signals[i] = 0.0
                    position = 0
                    continue
                if curr_close < ema50_4h_aligned[i]:  # price below trend
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: EMA12 crosses above EMA26 OR price rises above 4h EMA50
            elif position == -1:
                if curr_ema12 > curr_ema26 and prev_ema12 <= prev_ema26:  # bullish crossover
                    signals[i] = 0.0
                    position = 0
                    continue
                if curr_close > ema50_4h_aligned[i]:  # price above trend
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: EMA crossover with volatility confirmation and trend filter
        if position == 0:
            # Bullish crossover: EMA12 crosses above EMA26
            bullish_crossover = curr_ema12 > curr_ema26 and prev_ema12 <= prev_ema26
            # Bearish crossover: EMA12 crosses below EMA26
            bearish_crossover = curr_ema12 < curr_ema26 and prev_ema12 >= prev_ema26
            
            # Long: bullish crossover AND price > 4h EMA50 AND ATR ratio > 1.8
            if bullish_crossover and curr_close > ema50_4h_aligned[i] and atr_ratio_aligned[i] > 1.8:
                signals[i] = 0.20
                position = 1
            # Short: bearish crossover AND price < 4h EMA50 AND ATR ratio > 1.8
            elif bearish_crossover and curr_close < ema50_4h_aligned[i] and atr_ratio_aligned[i] > 1.8:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_EMACrossover_4hEMA50_Trend_1dATR_VolumeSpike_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0