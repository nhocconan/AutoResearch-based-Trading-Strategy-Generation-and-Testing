#!/usr/bin/env python3
"""
Experiment #8699: 6h Donchian breakout + 12h Camarilla pivot + volume confirmation + ATR stoploss.
Hypothesis: Combines price channel breakouts with institutional pivot levels on higher timeframe.
6h timeframe balances responsiveness with lower trade frequency. 12h Camarilla levels provide
support/resistance zones for filtering breakouts. Volume confirmation ensures institutional
participation. ATR stops manage risk. Designed to work in both bull (breakouts) and bear (fades at S3/R3).
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

from mtf_data import get_align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8699_6h_donchian20_12h_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
CAMARILLA_PERIOD = 20
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
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    c = close + (range_val * 1.1 / 12)
    d = close - (range_val * 1.1 / 12)
    l3 = close + (range_val * 1.1 / 6)
    h3 = close - (range_val * 1.1 / 6)
    l4 = close + (range_val * 1.1 / 4)
    h4 = close - (range_val * 1.1 / 4)
    return h4, l4, h3, l3, c, d

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Initialize Camarilla arrays
    h4_12h = np.full_like(close_12h, np.nan)
    l4_12h = np.full_like(close_12h, np.nan)
    h3_12h = np.full_like(close_12h, np.nan)
    l3_12h = np.full_like(close_12h, np.nan)
    
    # Calculate Camarilla for each 12h bar
    for i in range(len(close_12h)):
        h4, l4, h3, l3, _, _ = calculate_camarilla(high_12h[i], low_12h[i], close_12h[i])
        h4_12h[i] = h4
        l4_12h[i] = l4
        h3_12h[i] = h3
        l3_12h[i] = l3
    
    # Camarilla breakout levels: H4 = bullish breakout, L4 = bearish breakdown
    h4_12h_aligned = align_htf_to_ltf(prices, df_12h, h4_12h)
    l4_12h_aligned = align_htf_to_ltf(prices, df_12h, l4_12h)
    
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
        if np.isnan(h4_12h_aligned[i]) or np.isnan(l4_12h_aligned[i]):
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
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Camarilla conditions: break above H4 = bullish, break below L4 = bearish
        camarilla_long = close[i] > h4_12h_aligned[i-1] if not np.isnan(h4_12h_aligned[i-1]) else False
        camarilla_short = close[i] < l4_12h_aligned[i-1] if not np.isnan(l4_12h_aligned[i-1]) else False
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = long_breakout and camarilla_long and volume_confirmed
        short_entry = short_breakout and camarilla_short and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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