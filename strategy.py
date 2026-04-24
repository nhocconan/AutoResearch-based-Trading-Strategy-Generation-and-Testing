#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume confirmation using 6h ATR spike.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA34 trend filter.
- Entry: Long when price breaks above Camarilla H3 level AND ATR ratio > 2.0 AND price > 12h EMA34.
         Short when price breaks below Camarilla L3 level AND ATR ratio > 2.0 AND price < 12h EMA34.
- Exit: Opposite Camarilla breakout (H4/L4) OR price crosses 12h EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 2.0 confirms significant volatility expansion to avoid false breakouts.
- 12h EMA34 provides trend filter to avoid counter-trend trades.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Camarilla levels derived from 1d OHLC provide institutional support/resistance levels.
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
    """Calculate Camarilla pivot levels."""
    range_ = high - low
    h5 = close + range_ * 1.1 / 2
    h4 = close + range_ * 1.1 / 4
    h3 = close + range_ * 1.1 / 6
    l3 = close - range_ * 1.1 / 6
    l4 = close - range_ * 1.1 / 4
    l5 = close - range_ * 1.1 / 2
    return h3, h4, h5, l3, l4, l5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h trend filter: EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    ema34_12h = ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h, additional_delay_bars=1)
    
    # Calculate 6h ATR for volume spike filter
    atr_20 = atr(high, low, close, 20)
    atr_current = atr(high, low, close, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    
    # Camarilla levels from 1d OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    camarilla_h3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    # Calculate Camarilla levels for each 1d bar and align to 6h
    for i in range(len(df_1d)):
        h3, h4, h5, l3, l4, l5 = camarilla_levels(
            df_1d['high'].iloc[i],
            df_1d['low'].iloc[i],
            df_1d['close'].iloc[i]
        )
        # Find 6h bars that belong to this 1d bar
        start_idx = i * 4  # 4x 6h bars in 1d
        end_idx = min(start_idx + 4, n)
        camarilla_h3[start_idx:end_idx] = h3
        camarilla_h4[start_idx:end_idx] = h4
        camarilla_l3[start_idx:end_idx] = l3
        camarilla_l4[start_idx:end_idx] = l4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(atr_ratio[i]) or
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 12h EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L4 OR price falls below 12h EMA34
            if position == 1:
                if curr_close < camarilla_l4[i] or curr_close < ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H4 OR price rises above 12h EMA34
            elif position == -1:
                if curr_close > camarilla_h4[i] or curr_close > ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND ATR ratio > 2.0 AND bullish 12h trend
            if curr_close > camarilla_h3[i] and atr_ratio[i] > 2.0 and curr_close > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND ATR ratio > 2.0 AND bearish 12h trend
            elif curr_close < camarilla_l3[i] and atr_ratio[i] > 2.0 and curr_close < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_6hATR_VolumeSpike_12hEMA34_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0