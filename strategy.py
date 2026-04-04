#!/usr/bin/env python3
"""
Experiment #2316: 12h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: 12h Donchian breakouts with 1d trend alignment and volume confirmation capture medium-term trends.
- Primary: 12h Donchian(20) breakout (long: close > highest high of prior 20 bars; short: close < lowest low of prior 20 bars)
- HTF: 1d HMA(21) trend filter (only trade in direction of 1d HMA slope)
- Volume: Require > 1.5x 20-bar average volume spike to confirm participation
- Exit: ATR(14) stoploss (2*ATR) or opposite Donchian channel touch
- Target: 50-150 total trades over 4 years (12-37/year) - suitable for 12h timeframe
- Works in bull markets (breakouts during uptrends) and bear markets (breakouts during downtrends)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2316_12h_donchian20_1d_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d HMA(21): HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    def hma(values, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        if half_period < 1 or sqrt_period < 1:
            return np.full_like(values, np.nan)
        wma_half = np.array([np.nan] * (len(values) - half_period + 1))
        wma_full = np.array([np.nan] * (len(values) - period + 1))
        for i in range(len(values) - half_period + 1):
            wma_half[i] = wma(values[i:i + half_period], half_period)
        for i in range(len(values) - period + 1):
            wma_full[i] = wma(values[i:i + period], period)
        raw_hma = 2 * wma_half - wma_full[:len(wma_half)]
        hma_values = np.array([np.nan] * (len(values) - sqrt_period + 1))
        for i in range(len(raw_hma)):
            hma_values[i] = wma(raw_hma[i:i + sqrt_period], sqrt_period)
        # Pad to original length
        result = np.full_like(values, np.nan)
        result[period - 1:] = hma_values
        return result
    
    hma_1d = hma(close_1d, 21)
    # Trend: 1 if HMA rising, -1 if falling
    hma_diff = np.diff(hma_1d, prepend=np.nan)
    trend_1d = np.where(hma_diff > 0, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 12h Indicators: Donchian(20), ATR(14), Volume MA(20) ===
    # Donchian channels: highest high and lowest low of prior 20 bars
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 14, 20) + 5  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches or crosses below lowest low of prior 20 bars (opposite Donchian)
                elif price <= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches or crosses above highest high of prior 20 bars (opposite Donchian)
                elif price >= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d HMA trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike and not np.isnan(trend_bias):
            # Long entry: price breaks above highest high of prior 20 bars with uptrend
            if trend_bias > 0 and price > highest_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lowest low of prior 20 bars with downtrend
            elif trend_bias < 0 and price < lowest_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals