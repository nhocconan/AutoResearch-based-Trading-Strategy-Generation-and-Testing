#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and 12h EMA34 trend filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for Camarilla pivot levels and ATR volume spike, 12h for EMA34 trend filter.
- Entry: Long when price breaks above Camarilla R3 AND 1d ATR ratio > 2.0 AND price > 12h EMA34.
         Short when price breaks below Camarilla S3 AND 1d ATR ratio > 2.0 AND price < 12h EMA34.
- Exit: Opposite Camarilla breakout (R4/S4) OR price crosses 12h EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 2.0 confirms significant volatility expansion to avoid false breakouts.
- 12h EMA34 provides trend filter to avoid counter-trend trades.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~120 total over 4 years (~30/year) based on volatility breakout frequency with strict filters.
- Camarilla levels use 1d OHLC to calculate intraday support/resistance levels proven effective in crypto.
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
    """
    Calculate Camarilla pivot levels for intraday trading.
    Based on previous day's OHLC.
    Levels: H5, H4, H3, L3, L4, L5
    R3 = Close + (High - Low) * 1.1 / 4
    S3 = Close - (High - Low) * 1.1 / 4
    R4 = Close + (High - Low) * 1.1 / 2
    S4 = Close - (High - Low) * 1.1 / 2
    """
    range_ = high - low
    r3 = close + range_ * 1.1 / 4
    s3 = close - range_ * 1.1 / 4
    r4 = close + range_ * 1.1 / 2
    s4 = close - range_ * 1.1 / 2
    return r3, s3, r4, s4

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
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate 1d Camarilla levels from previous day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Shift OHLC by 1 to use previous day's levels (no look-ahead)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    # First value will be NaN due to roll, handled by min_periods in alignment
    r3, s3, r4, s4 = camarilla_levels(prev_high, prev_low, prev_close)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3, additional_delay_bars=1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4, additional_delay_bars=1)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 12h EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below S4 OR price falls below 12h EMA34
            if position == 1:
                if curr_close < s4_aligned[i] or curr_close < ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above R4 OR price rises above 12h EMA34
            elif position == -1:
                if curr_close > r4_aligned[i] or curr_close > ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above R3 AND ATR ratio > 2.0 AND bullish 12h trend
            if curr_close > r3_aligned[i] and atr_ratio_aligned[i] > 2.0 and curr_close > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND ATR ratio > 2.0 AND bearish 12h trend
            elif curr_close < s3_aligned[i] and atr_ratio_aligned[i] > 2.0 and curr_close < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dATR_VolumeSpike_12hEMA34_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0