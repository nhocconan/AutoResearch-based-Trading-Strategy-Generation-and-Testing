#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d ATR volume spike filter targeting 60-150 total trades over 4 years (15-37/year).
- Primary timeframe: 1h using HTF 4h for trend direction and 1d for volatility confirmation.
- Entry: Long when price breaks above Camarilla R1 AND ATR ratio > 2.0 AND price > 4h EMA50.
         Short when price breaks below Camarilla S1 AND ATR ratio > 2.0 AND price < 4h EMA50.
- Exit: Opposite Camarilla breakout OR price crosses 4h EMA50 in opposite direction.
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-volume periods.
- Camarilla levels derived from prior 1h OHLC provide intraday support/resistance.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with strict filters.
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

def camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels (R1, S1)."""
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    r1 = pivot + range_hl * 1.1 / 12.0
    s1 = pivot - range_hl * 1.1 / 12.0
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Precompute session hours (08:00-20:00 UTC) using DatetimeIndex
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
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
    
    # Camarilla levels from prior 1h OHLC
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Prior hour's OHLC (index i-1 corresponds to prior completed hour)
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        r1, s1 = camarilla_levels(ph, pl, pc)
        camarilla_r1[i] = r1
        camarilla_s1[i] = s1
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 4h EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla S1 OR price falls below 4h EMA50
            if position == 1:
                if curr_close < camarilla_s1[i] or curr_close < ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla R1 OR price rises above 4h EMA50
            elif position == -1:
                if curr_close > camarilla_r1[i] or curr_close > ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla R1 AND ATR ratio > 2.0 AND bullish 4h trend
            if curr_close > camarilla_r1[i] and atr_ratio_aligned[i] > 2.0 and curr_close > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S1 AND ATR ratio > 2.0 AND bearish 4h trend
            elif curr_close < camarilla_s1[i] and atr_ratio_aligned[i] > 2.0 and curr_close < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_TrendFilter_1dATR_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0