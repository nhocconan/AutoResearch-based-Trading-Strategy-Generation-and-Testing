#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d ATR volume spike and 1d ADX trend filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ATR volume spike and ADX trend filter.
- Entry: Long when price breaks above Camarilla H3 level AND ATR ratio > 1.8 AND ADX > 25.
         Short when price breaks below Camarilla L3 level AND ATR ratio > 1.8 AND ADX > 25.
- Exit: Opposite Camarilla breakout (L3 for longs, H3 for shorts).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide intraday support/resistance based on prior day's range.
- ATR ratio confirms volatility expansion to avoid false breakouts in low-volume periods.
- ADX > 25 ensures we only trade in trending markets, reducing whipsaws in ranging conditions.
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

def adx(high, low, close, period):
    """Calculate Average Directional Index."""
    # Calculate True Range
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth the values
    tr_smoothed = pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smoothed = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smoothed = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate Directional Indicators
    plus_di = 100 * plus_dm_smoothed / (tr_smoothed + 1e-10)
    minus_di = 100 * minus_dm_smoothed / (tr_smoothed + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_values = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx_values

def camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels (H3, L3, H4, L4)."""
    range_val = high - low
    h3 = close + (range_val * 1.1 / 4)
    l3 = close - (range_val * 1.1 / 4)
    h4 = close + (range_val * 1.1 / 2)
    l4 = close - (range_val * 1.1 / 2)
    return h3, l3, h4, l4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate 1d ADX for trend filter
    adx_14 = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for ATR and ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels using previous day's OHLC
        if i < 1:
            camarilla_ready = False
        else:
            camarilla_ready = True
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
        
        if camarilla_ready:
            camarilla_h3, camarilla_l3, _, _ = camarilla_levels(prev_high, prev_low, prev_close)
        else:
            camarilla_h3 = np.nan
            camarilla_l3 = np.nan
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout
        if position != 0:
            # Exit long: price breaks below Camarilla L3
            if position == 1:
                if curr_close < camarilla_l3:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3
            elif position == -1:
                if curr_close > camarilla_h3:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0 and camarilla_ready:
            # Long: price breaks above Camarilla H3 AND ATR ratio > 1.8 AND ADX > 25
            if curr_close > camarilla_h3 and atr_ratio_aligned[i] > 1.8 and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND ATR ratio > 1.8 AND ADX > 25
            elif curr_close < camarilla_l3 and atr_ratio_aligned[i] > 1.8 and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_1dADX_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0