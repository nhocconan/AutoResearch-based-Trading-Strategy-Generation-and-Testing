#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h EMA34 trend filter and volume confirmation using ATR spike.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA34 trend filter and 1d for ATR spike volume confirmation.
- Entry: Long when price breaks above Camarilla H3 level AND ATR ratio > 1.5 AND price > 4h EMA34.
         Short when price breaks below Camarilla L3 level AND ATR ratio > 1.5 AND price < 4h EMA34.
- Exit: Opposite Camarilla breakout OR price crosses 4h EMA34 in opposite direction.
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/14-period ATR) > 1.5 confirms significant volatility expansion to avoid false breakouts.
- 4h EMA34 provides trend filter to avoid counter-trend trades.
- Session filter: 08-20 UTC to avoid low-liquidity periods.
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

def camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels."""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    H3 = pivot + (range_ * 1.1 / 4)
    L3 = pivot - (range_ * 1.1 / 4)
    return H3, L3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h trend filter: EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 40:
        return np.zeros(n)
    
    ema34_4h = ema(df_4h['close'].values, 34)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    atr_14 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_14 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for current bar
        H3, L3 = camarilla_pivot(high[i], low[i], close[i])
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 4h EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below L3 OR price falls below 4h EMA34
            if position == 1:
                if curr_close < L3 or curr_close < ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above H3 OR price rises above 4h EMA34
            elif position == -1:
                if curr_close > H3 or curr_close > ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above H3 AND ATR ratio > 1.5 AND bullish 4h trend
            if curr_close > H3 and atr_ratio_aligned[i] > 1.5 and curr_close > ema34_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below L3 AND ATR ratio > 1.5 AND bearish 4h trend
            elif curr_close < L3 and atr_ratio_aligned[i] > 1.5 and curr_close < ema34_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_TrendFilter_1dATR_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0