#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d ATR volume spike filter and 1d EMA34 trend filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and ATR volume spike confirmation.
- Entry: Long when price breaks above Camarilla H3 level AND ATR ratio > 1.8 AND price > 1d EMA34.
         Short when price breaks below Camarilla L3 level AND ATR ratio > 1.8 AND price < 1d EMA34.
- Exit: Opposite Camarilla breakout (H4/L4) OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 1.8 confirms significant volatility expansion to avoid false breakouts.
- 1d EMA34 provides trend filter to avoid counter-trend trades.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with strict filters.
- Camarilla levels calculated from prior 1d OHLC (H3/L3 = close + 1.1*(high-low)/6, H4/L4 = close + 1.5*(high-low)/6).
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
    H3 = close + range_ * 1.1 / 6
    L3 = close - range_ * 1.1 / 6
    H4 = close + range_ * 1.5 / 6
    L4 = close - range_ * 1.5 / 6
    return H3, L3, H4, L4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate Camarilla levels from prior 1d OHLC (using 1d data)
    camarilla_H3, camarilla_L3, camarilla_H4, camarilla_L4 = camarilla_levels(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3, additional_delay_bars=1)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3, additional_delay_bars=1)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4, additional_delay_bars=1)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout (H4/L4) OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L4 OR price falls below 1d EMA34
            if position == 1:
                if curr_close < camarilla_L4_aligned[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H4 OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > camarilla_H4_aligned[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND ATR ratio > 1.8 AND bullish 1d trend
            if curr_close > camarilla_H3_aligned[i] and atr_ratio_aligned[i] > 1.8 and curr_close > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND ATR ratio > 1.8 AND bearish 1d trend
            elif curr_close < camarilla_L3_aligned[i] and atr_ratio_aligned[i] > 1.8 and curr_close < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_1dEMA34_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0