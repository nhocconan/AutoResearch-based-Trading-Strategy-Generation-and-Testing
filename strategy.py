#!/usr/bin/env python3
"""
exp_6955_6h_donchian20_1w_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with weekly Camarilla pivot continuation (R4/S4) and volume confirmation.
In bull markets (price > weekly R4): long breakouts only. In bear markets (price < weekly S4): short breakouts only.
Weekly Camarilla pivots provide structural levels to avoid fakeouts. Volume confirms breakout legitimacy.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Uses 1w HTF for pivot calculation to reduce noise and increase reliability.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6955_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 20  # ~5 days (6h bars)
PIVOT_LOOKBACK = 5  # bars to confirm pivot level respect

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for weekly Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivots (based on prior week's OHLC)
    # Need to shift by 1 to avoid look-ahead (use completed week only)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivots from PREVIOUS week (shifted by 1)
    # We'll calculate for each bar using the most recent completed weekly bar
    pivots_high = np.full_like(close_1w, np.nan)
    pivots_low = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        # Use previous week's OHLC to calculate current week's pivots
        phigh = high_1w[i-1]
        plow = low_1w[i-1]
        pclose = close_1w[i-1]
        pivot = (phigh + plow + pclose) / 3
        range_ = phigh - plow
        # Camarilla levels
        r4 = pclose + range_ * 1.1 / 2
        s4 = pclose - range_ * 1.1 / 2
        pivots_high[i] = r4
        pivots_low[i] = s4
    
    # Align to LTF (6h)
    pivots_high_aligned = align_htf_to_ltf(prices, df_1w, pivots_high)
    pivots_low_aligned = align_htf_to_ltf(prices, df_1w, pivots_low)
    
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
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 5  # extra for pivot calc
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(pivots_high_aligned[i]) or np.isnan(pivots_low_aligned[i]):
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
        
        # Determine market regime from weekly Camarilla
        # Bull: price above weekly R4, Bear: price below weekly S4
        weekly_bull = close[i] > pivots_high_aligned[i]
        weekly_bear = close[i] < pivots_low_aligned[i]
        
        # Breakout signals aligned with weekly regime
        long_breakout = weekly_bull and (close[i] > highest_high[i]) and vol_confirmed
        short_breakout = weekly_bear and (close[i] < lowest_low[i]) and vol_confirmed
        
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