#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume confirmation using 12h ATR spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend filter and 12h for ATR volume spike filter.
- Entry: Long when price breaks above Camarilla H3 AND ATR ratio > 1.8 AND price > 1d EMA50.
         Short when price breaks below Camarilla L3 AND ATR ratio > 1.8 AND price < 1d EMA50.
- Exit: Opposite Camarilla breakout OR price crosses 1d EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 1.8 confirms significant volatility expansion to avoid false breakouts.
- 1d EMA50 provides trend filter to avoid counter-trend trades.
- Camarilla levels derived from prior 1d OHLC provide institutional support/resistance.
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
    """Calculate Camarilla pivot levels (H3, L3)."""
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    h3 = pivot + range_hl * 1.1 / 6.0
    l3 = pivot - range_hl * 1.1 / 6.0
    return h3, l3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Calculate 12h ATR for volume spike filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    atr_20_12h = atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 20)
    atr_current_12h = atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 1)
    atr_ratio_12h = atr_current_12h / (atr_20_12h + 1e-10)  # Avoid division by zero
    atr_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio_12h, additional_delay_bars=1)
    
    # Camarilla levels from prior 1d OHLC
    df_1d_camarilla = get_htf_data(prices, '1d')
    if len(df_1d_camarilla) < 2:
        return np.zeros(n)
    
    # Use prior day's OHLC for today's Camarilla levels (no look-ahead)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Prior day's OHLC (index i-1 in 1d data corresponds to prior completed day)
        prior_day_idx = i - 1
        if prior_day_idx < len(df_1d_camarilla):
            ph = df_1d_camarilla['high'].iloc[prior_day_idx]
            pl = df_1d_camarilla['low'].iloc[prior_day_idx]
            pc = df_1d_camarilla['close'].iloc[prior_day_idx]
            h3, l3 = camarilla_levels(ph, pl, pc)
            camarilla_h3[i] = h3
            camarilla_l3[i] = l3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 70  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_ratio_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1d EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L3 OR price falls below 1d EMA50
            if position == 1:
                if curr_close < camarilla_l3[i] or curr_close < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3 OR price rises above 1d EMA50
            elif position == -1:
                if curr_close > camarilla_h3[i] or curr_close > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND ATR ratio > 1.8 AND bullish 1d trend
            if curr_close > camarilla_h3[i] and atr_ratio_12h_aligned[i] > 1.8 and curr_close > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND ATR ratio > 1.8 AND bearish 1d trend
            elif curr_close < camarilla_l3[i] and atr_ratio_12h_aligned[i] > 1.8 and curr_close < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA50_TrendFilter_12hATR_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0