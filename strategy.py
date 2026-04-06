#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot levels with daily trend filter and volume confirmation.
# Uses pivot points from previous day for mean reversion in ranging markets and breakout in trending.
# Daily EMA filter ensures alignment with higher timeframe trend.
# Volume confirms institutional participation.
# Works in both bull and bear markets by adapting to regime via trend filter.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13576_12h_camarilla1d_volume_ema_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_LOOKBACK = 1  # Use previous day's OHLC
EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    multiplier = 1.1 / 12
    c = close
    h4 = c + range_val * multiplier * 11/2
    h3 = c + range_val * multiplier * 11/4
    h2 = c + range_val * multiplier * 11/6
    l2 = c - range_val * multiplier * 11/6
    l3 = c - range_val * multiplier * 11/4
    l4 = c - range_val * multiplier * 11/2
    return h3, h2, l3, l2  # Return key levels: H3, H2, L3, L2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate previous day's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get previous day's levels (avoid look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # Set first day's values to NaN (no previous day)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    camarilla_h3 = np.full_like(close_1d, np.nan)
    camarilla_h2 = np.full_like(close_1d, np.nan)
    camarilla_l3 = np.full_like(close_1d, np.nan)
    camarilla_l2 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        h3, h2, l3, l2 = calculate_camarilla(high_1d_prev[i], low_1d_prev[i], close_1d_prev[i])
        camarilla_h3[i] = h3
        camarilla_h2[i] = h2
        camarilla_l3[i] = l3
        camarilla_l2[i] = l2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (np.nanmean(volume[max(0, i-VOLUME_MA_PERIOD+1):i+1]) * VOLUME_THRESHOLD) if i >= VOLUME_MA_PERIOD else False
        
        # Trend filter: price above/below daily EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Camarilla-based signals
        # Long when price touches L3 support in uptrend or breaks H3 resistance
        # Short when price touches H3 resistance in downtrend or breaks L3 support
        camarilla_long = (volume_ok and 
                         ((uptrend and low[i] <= camarilla_l3_aligned[i]) or 
                          (not uptrend and high[i] >= camarilla_h3_aligned[i])))
        camarilla_short = (volume_ok and 
                          ((downtrend and high[i] >= camarilla_h3_aligned[i]) or 
                           (not downtrend and low[i] <= camarilla_l3_aligned[i])))
        
        # Generate signals
        if position == 0:
            if camarilla_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif camarilla_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals