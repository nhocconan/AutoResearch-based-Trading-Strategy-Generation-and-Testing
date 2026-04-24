#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d ATR volatility filter and 1w EMA200 trend filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ATR volatility confirmation, 1w for EMA200 trend filter.
- Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
- Entry: Long when Bull Power > 0 AND ATR ratio > 1.5 AND price > 1w EMA200.
         Short when Bear Power < 0 AND ATR ratio > 1.5 AND price < 1w EMA200.
- Exit: Opposite Elder Ray signal (Bull Power < 0 for long exit, Bear Power > 0 for short exit) OR price crosses 1w EMA200.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 1.5 confirms significant volatility expansion to avoid false signals.
- 1w EMA200 provides strong trend filter to avoid counter-trend trades in choppy markets.
- Works in bull markets (buy strength in uptrend) and bear markets (sell weakness in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility-filtered Elder Ray signals.
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
    
    # Calculate 1w trend filter: EMA200
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 210:
        return np.zeros(n)
    
    ema200_1w = ema(df_1w['close'].values, 200)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w, additional_delay_bars=1)
    
    # Calculate 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate EMA13 for Elder Ray (on primary 6h timeframe)
    ema13 = ema(close, 13)
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for 1w EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Elder Ray signal OR price crosses 1w EMA200
        if position != 0:
            # Exit long: Bull Power turns negative OR price falls below 1w EMA200
            if position == 1:
                if bull_power[i] < 0 or curr_close < ema200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bear Power turns positive OR price rises above 1w EMA200
            elif position == -1:
                if bear_power[i] > 0 or curr_close > ema200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Elder Ray signal with volatility confirmation and trend filter
        if position == 0:
            # Long: Bull Power > 0 AND ATR ratio > 1.5 AND bullish 1w trend
            if bull_power[i] > 0 and atr_ratio_aligned[i] > 1.5 and curr_close > ema200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND ATR ratio > 1.5 AND bearish 1w trend
            elif bear_power[i] < 0 and atr_ratio_aligned[i] > 1.5 and curr_close < ema200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dATR_VolumeSpike_1wEMA200_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0