#!/usr/bin/env python3
"""
Hypothesis: Daily Williams %R extreme reversal with 1-week EMA34 trend filter and 1d ATR volume spike confirmation.
- Primary timeframe: 1d targeting 30-80 total trades over 4 years (7-20/year).
- HTF: 1w for EMA34 trend filter.
- Entry: Long when Williams %R(14) crosses above -80 (oversold reversal) AND ATR ratio > 1.5 AND price > 1w EMA34.
         Short when Williams %R(14) crosses below -20 (overbought reversal) AND ATR ratio > 1.5 AND price < 1w EMA34.
- Exit: Opposite Williams %R extreme OR price crosses 1w EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 1.5 confirms significant volatility expansion to avoid false reversals.
- 1w EMA34 provides trend filter to avoid counter-trend trades.
- Works in bull markets (buy oversold reversals in uptrend) and bear markets (sell overbought reversals in downtrend).
- Estimated trades: ~50 total over 4 years (~12/year) based on extreme reversal frequency with strict filters.
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

def williams_r(high, low, close, period):
    """Calculate Williams %R."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    ema34_1w = ema(df_1w['close'].values, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Williams %R on 1d (14-period)
    wr = williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(wr[i]) or np.isnan(ema34_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_wr = wr[i]
        prev_wr = wr[i-1] if i > 0 else -50  # Previous Williams %R
        
        # Exit conditions: opposite Williams %R extreme OR price crosses 1w EMA34 in opposite direction
        if position != 0:
            # Exit long: Williams %R rises above -20 (overbought) OR price falls below 1w EMA34
            if position == 1:
                if curr_wr > -20 or curr_close < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R falls below -80 (oversold) OR price rises above 1w EMA34
            elif position == -1:
                if curr_wr < -80 or curr_close > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme reversal with volatility confirmation and trend filter
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND ATR ratio > 1.5 AND bullish 1w trend
            if prev_wr <= -80 and curr_wr > -80 and atr_ratio_aligned[i] > 1.5 and curr_close > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND ATR ratio > 1.5 AND bearish 1w trend
            elif prev_wr >= -20 and curr_wr < -20 and atr_ratio_aligned[i] > 1.5 and curr_close < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_ExtremeReversal_1dATR_VolumeSpike_1wEMA34_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0