#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot levels with daily volume confirmation and weekly EMA trend filter.
# Uses Camarilla levels (H4/L4) from daily timeframe for precise reversal entries in ranging markets.
# Weekly EMA filter ensures trades align with higher timeframe trend. Volume confirmation avoids false breakouts.
# Works in ranging markets (mean reversion at H4/L4) and trending markets (breakouts beyond H5/L5).
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13512_12h_camarilla_1d_vol_1w_ema_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
EMA_PERIOD = 21
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
    h4 = close + range_val * multiplier * 11/2
    l4 = close - range_val * multiplier * 11/2
    h5 = close + range_val * multiplier * 11/1
    l5 = close - range_val * multiplier * 11/1
    return h4, l4, h5, l5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    camarilla_h5 = np.full(len(close_1d), np.nan)
    camarilla_l5 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        h4, l4, h5, l5 = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        camarilla_h4[i] = h4
        camarilla_l4[i] = l4
        camarilla_h5[i] = h5
        camarilla_l5[i] = l5
    
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Load weekly EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if data not available
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Camarilla-based signals
        # Long setup: price rejects L4 (support) in uptrend or breaks above H5 (strong breakout)
        long_setup = (volume_ok and 
                     ((close[i] > camarilla_l4_aligned[i] and low[i] <= camarilla_l4_aligned[i]) or  # bounce off L4
                      (high[i] > camarilla_h5_aligned[i] and uptrend)))  # break above H5 in uptrend
        
        # Short setup: price rejects H4 (resistance) in downtrend or breaks below L5 (strong breakdown)
        short_setup = (volume_ok and 
                      ((close[i] < camarilla_h4_aligned[i] and high[i] >= camarilla_h4_aligned[i]) or  # rejection at H4
                       (low[i] < camarilla_l5_aligned[i] and downtrend)))  # break below L5 in downtrend
        
        # Generate signals
        if position == 0:
            if long_setup:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_setup:
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