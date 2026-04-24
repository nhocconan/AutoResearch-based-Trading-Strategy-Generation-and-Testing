#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation using 12h ATR spike.
- Primary timeframe: 4h targeting 100-200 total trades over 4 years (25-50/year).
- HTF: 12h for EMA50 trend filter and ATR volume spike.
- Entry: Long when price breaks above Camarilla R1 AND ATR ratio > 1.8 AND price > 12h EMA50.
         Short when price breaks below Camarilla S1 AND ATR ratio > 1.8 AND price < 12h EMA50.
- Exit: Opposite Camarilla breakout OR price crosses 12h EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 1.8 confirms significant volatility expansion to avoid false breakouts.
- 12h EMA50 provides trend filter to avoid counter-trend trades.
- Camarilla levels provide intraday support/resistance with statistical edge.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~150 total over 4 years (~38/year) based on volatility breakout frequency with strict filters.
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

def camarilla_pivot(high, low, close):
    """Calculate Camarilla Pivot levels."""
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Calculate 12h ATR for volume spike filter
    if len(df_12h) < 30:
        return np.zeros(n)
    
    atr_20_12h = atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 20)
    atr_current_12h = atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 1)
    atr_ratio_12h = atr_current_12h / (atr_20_12h + 1e-10)  # Avoid division by zero
    atr_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio_12h, additional_delay_bars=1)
    
    # Calculate Camarilla levels on 4h
    camarilla_r1 = np.zeros(n)
    camarilla_s1 = np.zeros(n)
    
    for i in range(n):
        r1, s1 = camarilla_pivot(high[i], low[i], close[i])
        camarilla_r1[i] = r1
        camarilla_s1[i] = s1
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(atr_ratio_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 12h EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla S1 OR price falls below 12h EMA50
            if position == 1:
                if curr_close < camarilla_s1[i] or curr_close < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla R1 OR price rises above 12h EMA50
            elif position == -1:
                if curr_close > camarilla_r1[i] or curr_close > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla R1 AND ATR ratio > 1.8 AND bullish 12h trend
            if curr_close > camarilla_r1[i] and atr_ratio_12h_aligned[i] > 1.8 and curr_close > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 AND ATR ratio > 1.8 AND bearish 12h trend
            elif curr_close < camarilla_s1[i] and atr_ratio_12h_aligned[i] > 1.8 and curr_close < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hATR_VolumeSpike_12hEMA50_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0