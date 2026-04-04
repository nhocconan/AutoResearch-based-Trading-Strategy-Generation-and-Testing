#!/usr/bin/env python3
"""
exp_6767_6h_donchian20_1d_pivot_v1
Hypothesis: 6h Donchian(20) breakout filtered by 1d Camarilla pivot levels. 
In trending markets (price outside H3/L3): breakout continuation trades. 
In ranging markets (price between H3/L3): fade at H4/L4 levels. 
Uses 1d Camarilla pivots calculated from prior day's OHLC, aligned to 6h bars.
Volume confirmation avoids false breakouts. Designed for 6h timeframe to capture 
medium-term swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to volatility regime via pivot levels.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6767_6h_donchian20_1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 10  # ~2.5 days (6h bars)
PIVOT_LOOKBACK = 1  # use prior day's OHLC for today's pivot

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels from prior day's OHLC
    # Camarilla formulas: 
    # H4 = close + 1.5*(high - low)
    # H3 = close + 1.0*(high - low)
    # L3 = close - 1.0*(high - low)
    # L4 = close - 1.5*(high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use prior day's OHLC (no look-ahead)
    high_1d_prior = np.roll(high_1d, 1)
    low_1d_prior = np.roll(low_1d, 1)
    close_1d_prior = np.roll(close_1d, 1)
    # First value will be NaN after roll, that's fine
    
    pivot_range = high_1d_prior - low_1d_prior
    h4 = close_1d_prior + 1.5 * pivot_range
    h3 = close_1d_prior + 1.0 * pivot_range
    l3 = close_1d_prior - 1.0 * pivot_range
    l4 = close_1d_prior - 1.5 * pivot_range
    
    # Align to LTF (6h)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
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
        
        # Skip if HTF data not available (first bar after roll)
        if np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]):
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
        
        # Determine market regime based on Camarilla levels
        # Trending: price outside H3/L3 zone
        # Ranging: price inside H3/L3 zone
        price_above_h3 = close[i] > h3_aligned[i]
        price_below_l3 = close[i] < l3_aligned[i]
        price_in_range = (close[i] >= l3_aligned[i]) & (close[i] <= h3_aligned[i])
        
        # Breakout signals
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # Fade signals at extreme levels
        long_fade = close[i] < l4_aligned[i]  # price at extreme low, mean reversion long
        short_fade = close[i] > h4_aligned[i]  # price at extreme high, mean reversion short
        
        # Entry logic: adaptive to regime
        if position == 0:
            if vol_confirmed:
                # In trending regime: trade breakouts
                if price_above_h3 or price_below_l3:
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
                # In ranging regime: fade extremes
                elif price_in_range:
                    if long_fade:
                        signals[i] = SIGNAL_SIZE
                        position = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                    elif short_fade:
                        signals[i] = -SIGNAL_SIZE
                        position = -1
                        entry_price = close[i]
                        bars_since_entry = 0
            # No volume confirmation: stay flat or hold
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals