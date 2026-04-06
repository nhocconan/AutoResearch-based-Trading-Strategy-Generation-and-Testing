#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour strategy using 12-hour Camarilla pivot levels with volume confirmation.
# Goes long when price breaks above R4 with volume, short when breaks below S4 with volume.
# Uses 12h trend (EMA50) as filter to avoid counter-trend trades.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.

name = "exp_13759_6h_camarilla_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC for pivot calculation
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 8
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Camarilla formulas
    pivot = (high + low + close) / 3
    range_ = high - low
    
    r4 = close + range_ * 1.1 / 2
    r3 = close + range_ * 1.1 / 4
    r2 = close + range_ * 1.1 / 6
    r1 = close + range_ * 1.1 / 12
    
    s1 = close - range_ * 1.1 / 12
    s2 = close - range_ * 1.1 / 6
    s3 = close - range_ * 1.1 / 4
    s4 = close - range_ * 1.1 / 2
    
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for Camarilla pivots and trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 12h Camarilla levels (using previous bar's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Shift by 1 to use previous bar's levels (avoid look-ahead)
    r1_12h = np.full_like(high_12h, np.nan)
    r2_12h = np.full_like(high_12h, np.nan)
    r3_12h = np.full_like(high_12h, np.nan)
    r4_12h = np.full_like(high_12h, np.nan)
    s1_12h = np.full_like(high_12h, np.nan)
    s2_12h = np.full_like(high_12h, np.nan)
    s3_12h = np.full_like(high_12h, np.nan)
    s4_12h = np.full_like(high_12h, np.nan)
    
    # Calculate Camarilla for each bar (using previous bar's OHLC)
    for i in range(1, len(high_12h)):
        r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high_12h[i-1], low_12h[i-1], close_12h[i-1])
        r1_12h[i] = r1
        r2_12h[i] = r2
        r3_12h[i] = r3
        r4_12h[i] = r4
        s1_12h[i] = s1
        s2_12h[i] = s2
        s3_12h[i] = s3
        s4_12h[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # 6h data for entry timing and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
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
        if np.isnan(ema_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or np.isnan(volume_ma[i]):
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
        
        # Trend direction from 12h EMA
        above_ema = close[i] > ema_12h_aligned[i]
        below_ema = close[i] < ema_12h_aligned[i]
        
        # Camarilla breakout signals
        long_signal = volume_ok and above_ema and close[i] > r4_12h_aligned[i]
        short_signal = volume_ok and below_ema and close[i] < s4_12h_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on close below R3 (mean reversion)
            if close[i] < r3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above S3 (mean reversion)
            if close[i] > s3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals