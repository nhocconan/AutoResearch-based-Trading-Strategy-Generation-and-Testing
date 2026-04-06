#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Camarilla pivot levels with volume confirmation on 12h timeframe
# Uses 12h bars for entries/exits, with daily Camarilla levels as support/resistance.
# Volume filter ensures only significant breakouts trigger trades.
# Works in bull/bear because pivot levels adapt to volatility and volume confirms strength.
# Target: 50-150 trades over 4 years (12-37/year) to balance opportunity and fee drag.

name = "exp_12916_12h_camarilla_pivot_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
VOLUME_MA_PERIOD = 20
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
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_ = high - low
    pivot = (high + low + close) / 3.0
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
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    r1_d = np.zeros(len(close_d))
    r2_d = np.zeros(len(close_d))
    r3_d = np.zeros(len(close_d))
    r4_d = np.zeros(len(close_d))
    s1_d = np.zeros(len(close_d))
    s2_d = np.zeros(len(close_d))
    s3_d = np.zeros(len(close_d))
    s4_d = np.zeros(len(close_d))
    
    for i in range(len(close_d)):
        r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high_d[i], low_d[i], close_d[i])
        r1_d[i] = r1
        r2_d[i] = r2
        r3_d[i] = r3
        r4_d[i] = r4
        s1_d[i] = s1
        s2_d[i] = s2
        s3_d[i] = s3
        s4_d[i] = s4
    
    # Align to 12h timeframe (each daily bar = 2 x 12h bars)
    r1_12h = align_htf_to_ltf(prices, df_daily, r1_d)
    r2_12h = align_htf_to_ltf(prices, df_daily, r2_d)
    r3_12h = align_htf_to_ltf(prices, df_daily, r3_d)
    r4_12h = align_htf_to_ltf(prices, df_daily, r4_d)
    s1_12h = align_htf_to_ltf(prices, df_daily, s1_d)
    s2_12h = align_htf_to_ltf(prices, df_daily, s2_d)
    s3_12h = align_htf_to_ltf(prices, df_daily, s3_d)
    s4_12h = align_htf_to_ltf(prices, df_daily, s4_d)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(r4_12h[i]) or np.isnan(s4_12h[i]):
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
        
        # Breakout above R4 or breakdown below S4
        breakout_long = volume_ok and close[i] >= r4_12h[i]
        breakout_short = volume_ok and close[i] <= s4_12h[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
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