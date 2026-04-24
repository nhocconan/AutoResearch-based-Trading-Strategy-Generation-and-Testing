#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1-week ADX trend filter and volume confirmation using 1d ATR spike.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for ADX trend filter (>25 = trending market).
- Entry: Long when price breaks above Camarilla H3 level AND ATR ratio > 1.8 AND 1w ADX > 25.
         Short when price breaks below Camarilla L3 level AND ATR ratio > 1.8 AND 1w ADX > 25.
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short) OR 1w ADX drops below 20 (trend ends).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels calculated from prior 1d OHLC: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6.
- ATR ratio (current ATR/20-period ATR) > 1.8 confirms significant volatility expansion to avoid false breakouts.
- 1w ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
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
    # True Range
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_period = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (atr_period + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (atr_period + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_vals = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    return adx_vals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w trend filter: ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    adx_1w = adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate Camarilla levels from prior 1d OHLC (need to shift by 1 to avoid look-ahead)
    # We use the prior day's OHLC to calculate today's levels
    prior_close = np.roll(close, 1)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    # Handle first bar
    prior_close[0] = close[0]
    prior_high[0] = high[0]
    prior_low[0] = low[0]
    
    cam_h3 = prior_close + 1.1 * (prior_high - prior_low) / 6
    cam_l3 = prior_close - 1.1 * (prior_high - prior_low) / 6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(cam_h3[i]) or np.isnan(cam_l3[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR 1w ADX drops below 20 (trend ends)
        if position != 0:
            # Exit long: price breaks below Camarilla L3 OR ADX < 20 (trend weakening)
            if position == 1:
                if curr_close < cam_l3[i] or adx_1w_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3 OR ADX < 20 (trend weakening)
            elif position == -1:
                if curr_close > cam_h3[i] or adx_1w_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND ATR ratio > 1.8 AND 1w ADX > 25 (strong trend)
            if curr_close > cam_h3[i] and atr_ratio_aligned[i] > 1.8 and adx_1w_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND ATR ratio > 1.8 AND 1w ADX > 25 (strong trend)
            elif curr_close < cam_l3[i] and atr_ratio_aligned[i] > 1.8 and adx_1w_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_1wADX_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0