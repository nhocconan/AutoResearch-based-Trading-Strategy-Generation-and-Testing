#!/usr/bin/env python3
"""
exp_7391_6h_donchian20_1d_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1d pivot-based trend filter and volume confirmation.
Uses daily high/low from previous day to establish trend (higher highs/lows = uptrend).
Volume breakout confirms strength. Designed for 60-120 trades over 4 years (15-30/year).
Works in bull/bear via price action pivot filter. Uses discrete sizing (0.25) to minimize fees.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7391_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BREAKOUT_MULT = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULT = 2.5
MAX_HOLD_BARS = 12

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for pivot trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d pivot points: trend based on higher highs/lows
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily trend: higher high and higher low = uptrend, lower both = downtrend
    hh = high_1d > np.roll(high_1d, 1)  # higher high than previous day
    hl = low_1d > np.roll(low_1d, 1)    # higher low than previous day
    lh = high_1d < np.roll(high_1d, 1)  # lower high
    ll = low_1d < np.roll(low_1d, 1)    # lower low
    
    uptrend = hh & hl
    downtrend = lh & ll
    
    # Forward fill trend to handle equal days
    uptrend_series = pd.Series(uptrend).replace(False, np.nan).ffill().fillna(False).values
    downtrend_series = pd.Series(downtrend).replace(False, np.nan).ffill().fillna(False).values
    
    # Align to LTF (6h)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend_series)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend_series)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for breakout confirmation
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
        
        # Skip if HTF trend data not available
        if np.isnan(uptrend_aligned[i]) or np.isnan(downtrend_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULT * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULT * atr[i]:
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
            
        # Volume breakout confirmation
        vol_breakout = volume[i] > vol_ma[i] * VOL_BREAKOUT_MULT if not np.isnan(vol_ma[i]) else False
        
        # Determine trend from pivot structure
        is_uptrend = uptrend_aligned[i]
        is_downtrend = downtrend_aligned[i]
        
        # Breakout entries with volume confirmation
        breakout_long = is_uptrend and (close[i] > highest_high[i]) and vol_breakout
        breakout_short = is_downtrend and (close[i] < lowest_low[i]) and vol_breakout
        
        # Enter new positions only if flat
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_short:
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