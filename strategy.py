#!/usr/bin/env python3
"""
exp_6728_12h_donchian20_1w_hma_v1
Hypothesis: 12h Donchian(20) breakout with 1w HMA(21) trend filter and volume confirmation.
In bull markets: buy breakouts above upper Donchian when 1w HMA is rising.
In bear markets: sell breakdowns below lower Donchian when 1w HMA is falling.
Volume confirmation ensures breakout legitimacy. ATR-based stoploss limits drawdown.
Designed for 12h timeframe to capture long-term swings with ~12-37 trades/year.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6728_12h_donchian20_1w_hma_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 12  # ~6 days (12h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for HMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA(21)
    close_1w = df_1w['close'].values
    hma_1w = calculate_hma(close_1w, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Breakout signals with 1w HMA trend filter
        long_breakout = (close[i] > highest_high[i]) and vol_confirmed and (hma_1w_aligned[i] > hma_1w_aligned[i-1])
        short_breakout = (close[i] < lowest_low[i]) and vol_confirmed and (hma_1w_aligned[i] < hma_1w_aligned[i-1])
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

def calculate_hma(values, period):
    """Calculate Hull Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan)
    
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.zeros_like(values)
    for i in range(half_period, len(values)):
        wma_half[i] = np.nansum(values[i-half_period+1:i+1] * np.arange(1, half_period+1)) / (half_period * (half_period + 1) / 2)
    
    # WMA of full period
    wma_full = np.zeros_like(values)
    for i in range(period, len(values)):
        wma_full[i] = np.nansum(values[i-period+1:i+1] * np.arange(1, period+1)) / (period * (period + 1) / 2)
    
    # Raw HMA
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of sqrt period
    hma = np.zeros_like(values)
    for i in range(sqrt_period, len(values)):
        hma[i] = np.nansum(raw_hma[i-sqrt_period+1:i+1] * np.arange(1, sqrt_period+1)) / (sqrt_period * (sqrt_period + 1) / 2)
    
    return hma