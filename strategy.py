#!/usr/bin/env python3
"""
exp_7203_4h_donchian20_12h_volume_v1
Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation for BTC/ETH/SOL.
In trending markets: breakout continuation in breakout direction with volume spike.
Uses 12h timeframe for HTF volume filter to reduce noise and false breakouts.
Designed for 4h timeframe to capture medium-term moves with ~25-50 trades/year (100-200 total over 4 years).
Volume confirmation ensures breakouts have participation, reducing whipsaws in ranging markets.
Works in both bull and bear markets by requiring volume confirmation for entries.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7203_4h_donchian20_12h_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_SPIKE_THRESHOLD = 2.0  # Volume must be 2x MA to confirm breakout
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 20  # ~20 * 4h = ~3.3 days

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h volume moving average
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # Align 12h volume MA to 4h timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
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
        if np.isnan(vol_ma_12h_aligned[i]):
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
            
        # Volume confirmation: current volume > 12h volume MA * threshold
        vol_confirmed = volume[i] > vol_ma_12h_aligned[i] * VOL_SPIKE_THRESHOLD if not np.isnan(vol_ma_12h_aligned[i]) else False
        
        # Donchian breakout conditions
        breakout_long = close[i] > highest_high[i]
        breakout_short = close[i] < lowest_low[i]
        
        # Enter new positions only if flat and volume confirmed
        if position == 0:
            if breakout_long and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_short and vol_confirmed:
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