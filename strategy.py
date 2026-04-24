#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d ADX trend filter and 1d ATR volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX trend filter and ATR volume spike, 1w for regime context (optional).
- Entry: Long when Williams %R(14) crosses above -80 (oversold reversal) AND ADX(14) > 25 (trending market) AND ATR ratio > 1.5 (volatility expansion).
         Short when Williams %R(14) crosses below -20 (overbought reversal) AND ADX(14) > 25 AND ATR ratio > 1.5.
- Exit: Opposite Williams %R extreme (%R crosses below -80 for long, above -20 for short) OR ADX falls below 20 (trend weakening).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams %R identifies exhaustion points in trends, ADX filters for trending environments to avoid false reversals in chop,
  ATR spike confirms momentum behind the reversal.
- Works in bull markets (buy oversold pullbacks in uptrend) and bear markets (sell overbought bounces in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on reversal frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_r(high, low, close, period):
    """Calculate Williams %R."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr

def adx(high, low, close, period):
    """Calculate Average Directional Index."""
    # True Range
    tr1 = pd.Series(high).rolling(window=1).max() - pd.Series(low).rolling(window=1).min()
    tr2 = np.abs(pd.Series(high).rolling(window=1).max() - pd.Series(close).shift(1))
    tr3 = np.abs(pd.Series(low).rolling(window=1).min() - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    up_move = pd.Series(high).diff().values
    down_move = -pd.Series(low).diff().values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_values = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    return adx_values

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
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    wr_14 = williams_r(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    wr_14_aligned = align_htf_to_ltf(prices, df_1d, wr_14, additional_delay_bars=1)
    
    # Calculate 1d ADX (14-period) for trend filter
    adx_14 = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter (current ATR / 20-period ATR)
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(wr_14_aligned[i]) or np.isnan(adx_14_aligned[i]) or
            np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Exit conditions
        if position != 0:
            # Exit long: Williams %R crosses below -80 (reversal) OR ADX < 20 (trend weakening)
            if position == 1:
                if wr_14_aligned[i] < -80 or adx_14_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses above -20 (reversal) OR ADX < 20 (trend weakening)
            elif position == -1:
                if wr_14_aligned[i] > -20 or adx_14_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R reversal with trend and volatility confirmation
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND ADX > 25 AND ATR ratio > 1.5
            if wr_14_aligned[i] > -80 and wr_14_aligned[i-1] <= -80 and adx_14_aligned[i] > 25 and atr_ratio_aligned[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND ADX > 25 AND ATR ratio > 1.5
            elif wr_14_aligned[i] < -20 and wr_14_aligned[i-1] >= -20 and adx_14_aligned[i] > 25 and atr_ratio_aligned[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_ExtremeReversal_1dADX_TrendFilter_1dATR_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0