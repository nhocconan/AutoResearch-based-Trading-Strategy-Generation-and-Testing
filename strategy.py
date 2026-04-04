#!/usr/bin/env python3
"""
exp_6655_6h_donchian20_1w_pivot_v1
Hypothesis: 6h Donchian(20) breakout with 1-week Camarilla pivot direction filter and volume confirmation.
Uses 6h primary timeframe (target: 75-200 total trades over 4 years). 1-week Camarilla pivot provides
trend bias: price above weekly pivot favors longs, below favors shorts. Volume confirms breakout strength.
ATR-based stoploss limits downside. Discrete sizing (0.25) reduces fee churn. Designed to work in both
bull and bear markets by trading breakouts in direction of weekly pivot structure.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6655_6h_donchian20_1w_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 5  # weeks for pivot calculation
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 12  # ~3 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for Camarilla pivot
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1w + low_1w + close_1w) / 3.0
    # Calculate Camarilla levels
    r4 = pp + ((high_1w - low_1w) * 1.1 / 2.0)
    r3 = pp + ((high_1w - low_1w) * 1.1 / 4.0)
    s3 = pp - ((high_1w - low_1w) * 1.1 / 4.0)
    s4 = pp - ((high_1w - low_1w) * 1.1 / 2.0)
    
    # Align to LTF (6h) with shift(1) for completed bars only
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
            
        # Determine trend bias from weekly Camarilla pivot
        # Above weekly pivot = bullish bias, below = bearish bias
        bullish_bias = close[i] > pp_aligned[i]
        bearish_bias = close[i] < pp_aligned[i]
        
        # Long conditions: 
        # 1. Break above Donchian HIGH (breakout)
        # 2. Volume confirmation
        # 3. Bullish bias from weekly pivot
        long_breakout = close[i] > donchian_high[i-1]
        long_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions:
        # 1. Break below Donchian LOW (breakdown)
        # 2. Volume confirmation
        # 3. Bearish bias from weekly pivot
        short_breakout = close[i] < donchian_low[i-1]
        short_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout and long_volume and bullish_bias:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout and short_volume and bearish_bias:
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