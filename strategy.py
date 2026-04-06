#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot levels with daily volume spike and weekly trend filter.
# Uses Camarilla levels (H3/L3 for reversal, H4/L4 for breakout) from daily pivots.
# In ranging markets: fade touches of H3/L3 with volume confirmation.
# In trending markets: breakouts of H4/L4 with weekly trend alignment.
# Volume filter ensures institutional participation. Designed for 12h timeframe to avoid overtrading.

name = "exp_13552_12h_camarilla1d_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
EMA_PERIOD = 21

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
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
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    h4 = pivot + (range_val * 1.1 / 2)
    l4 = pivot - (range_val * 1.1 / 2)
    h3 = pivot + (range_val * 1.1 / 4)
    l3 = pivot - (range_val * 1.1 / 4)
    h2 = pivot + (range_val * 1.1 / 6)
    l2 = pivot - (range_val * 1.1 / 6)
    h1 = pivot + (range_val * 1.1 / 12)
    l1 = pivot - (range_val * 1.1 / 12)
    return h1, h2, h3, h4, l1, l2, l3, l4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    h1_1d, h2_1d, h3_1d, h4_1d, l1_1d, l2_1d, l3_1d, l4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 12h timeframe
    h1_1d_aligned = align_htf_to_ltf(prices, df_1d, h1_1d)
    h2_1d_aligned = align_htf_to_ltf(prices, df_1d, h2_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l1_1d_aligned = align_htf_to_ltf(prices, df_1d, l1_1d)
    l2_1d_aligned = align_htf_to_ltf(prices, df_1d, l2_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Load weekly EMA for trend filter ONCE before loop
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
        # Skip if any data not available
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Camarilla-based signals
        # Fade H3/L3 in ranging markets (price touches and reverses)
        fade_long = volume_ok and (low[i] <= l3_1d_aligned[i]) and (close[i] > l3_1d_aligned[i])
        fade_short = volume_ok and (high[i] >= h3_1d_aligned[i]) and (close[i] < h3_1d_aligned[i])
        
        # Breakout H4/L4 with trend alignment
        breakout_long = volume_ok and uptrend and (high[i] > h4_1d_aligned[i])
        breakout_short = volume_ok and downtrend and (low[i] < l4_1d_aligned[i])
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
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