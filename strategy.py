#!/usr/bin/env python3
"""
Experiment #8431: 6h Donchian breakout + 1d pivot direction + volume confirmation + ATR stoploss.
Hypothesis: 6-hour timeframe balances trend capture with reduced trade frequency.
Using 1-day Camarilla pivot levels: fade at R3/S3 (mean reversion in range), 
breakout continuation at R4/S4 (trend acceleration). Volume filter ensures institutional 
participation. Works in bull (breakouts) and bear (fades at pivot extremes).
Targets 50-150 trades over 4 years (12-37/year) to avoid fee drag.
"""

from mtf_data import get_alpho_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8431_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # Previous day for pivot calculation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivot(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_ = high - low
    r4 = close + range_ * 1.1 / 2
    r3 = close + range_ * 1.1 / 4
    s3 = close - range_ * 1.1 / 4
    s4 = close - range_ * 1.1 / 2
    return pivot, r3, r4, s3, s4

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = np.full(len(close_1d), np.nan)
    r3 = np.full(len(close_1d), np.nan)
    r4 = np.full(len(close_1d), np.nan)
    s3 = np.full(len(close_1d), np.nan)
    s4 = np.full(len(close_1d), np.nan)
    
    for i in range(PIVOT_LOOKBACK, len(close_1d)):
        p, r3_val, r4_val, s3_val, s4_val = calculate_pivot(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        pivot[i] = p
        r3[i] = r3_val
        r4[i] = r4_val
        s3[i] = s3_val
        s4[i] = s4_val
    
    # Pivot signals: 1 = fade at R3/S3 (mean reversion), 2 = breakout at R4/S4 (continuation)
    pivot_signal = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if not np.isnan(r3[i]) and not np.isnan(s3[i]):
            if close_1d[i] >= r3[i]:  # At or above R3
                pivot_signal[i] = -1  # Fade (expect reversal down)
            elif close_1d[i] <= s3[i]:  # At or below S3
                pivot_signal[i] = 1   # Fade (expect reversal up)
            elif close_1d[i] >= r4[i]:  # Break above R4
                pivot_signal[i] = 2   # Breakout continuation up
            elif close_1d[i] <= s4[i]:  # Break below S4
                pivot_signal[i] = -2  # Breakout continuation down
    
    pivot_signal_aligned = align_htf_to_ltf(prices, df_1d, pivot_signal)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(pivot_signal_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine market bias from 1d pivot
        fade_signal = pivot_signal_aligned[i] in [-1, 1]  # Fade at R3/S3
        breakout_signal = pivot_signal_aligned[i] in [-2, 2]  # Breakout at R4/S4
        
        # Fade logic: expect reversal from extreme
        fade_long = fade_signal and pivot_signal_aligned[i] == 1   # At S3, expect up
        fade_short = fade_signal and pivot_signal_aligned[i] == -1 # At R3, expect down
        
        # Breakout logic: expect continuation
        breakout_long = breakout_signal and pivot_signal_aligned[i] == 2   # Above R4, continue up
        breakout_short = breakout_signal and pivot_signal_aligned[i] == -2 # Below R4, continue down
        
        # Donchian breakout conditions (for breakout mode)
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        # Fade trades: counter-trend at pivot extremes
        fade_long_entry = fade_long and not volume_confirmed  # Fade works better on lower volume
        fade_short_entry = fade_short and not volume_confirmed
        
        # Breakout trades: trend continuation with volume
        breakout_long_entry = breakout_long and long_breakout and volume_confirmed
        breakout_short_entry = breakout_short and short_breakout and volume_confirmed
        
        # Generate signals
        if position == 0:
            if fade_long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short_entry:
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