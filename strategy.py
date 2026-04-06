#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining 1-day ATR breakout with volume confirmation.
# Goes long when price breaks above daily open + ATR(14) with volume (strong uptrend),
# short when breaks below daily open - ATR(14) with volume (strong downtrend).
# Uses 1-week trend (EMA50) as filter to avoid counter-trend trades.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.
# ATR breakouts capture volatility expansion while volume confirms institutional interest.

name = "exp_13772_12h_atrbreakout_1d_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.0
VOLUME_MA_PERIOD = 6
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
TREND_EMA_PERIOD = 50

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for daily open and ATR, and 1w data for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, TREND_EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d ATR for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    
    # Daily open (from 1d data)
    open_1d = df_1d['open'].values
    
    # Breakout levels: daily open ± ATR
    breakout_up = open_1d + (atr_1d * ATR_MULTIPLIER)
    breakout_down = open_1d - (atr_1d * ATR_MULTIPLIER)
    
    # Align breakout levels and EMA to 12h timeframe
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 12h data for entry timing and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss (using 12h data)
    atr_12h = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(breakout_up_aligned[i]) or np.isnan(breakout_down_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 1w EMA
        above_ema = close[i] > ema_1w_aligned[i]
        below_ema = close[i] < ema_1w_aligned[i]
        
        # ATR breakout signals
        long_signal = volume_ok and above_ema and close[i] > breakout_up_aligned[i]
        short_signal = volume_ok and below_ema and close[i] < breakout_down_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr_12h[i])  # 2x ATR stop
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr_12h[i])  # 2x ATR stop
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on close below daily open (mean reversion)
            if close[i] < open_1d[i]:  # Use 12h-aligned daily open
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above daily open (mean reversion)
            if close[i] > open_1d[i]:  # Use 12h-aligned daily open
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals