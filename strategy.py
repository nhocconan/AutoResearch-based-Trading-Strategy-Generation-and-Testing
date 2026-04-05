#!/usr/bin/env python3
"""
Experiment #9291: 6s Donchian breakout + 1d Camarilla pivot + volume confirmation.
Hypothesis: Donchian breakouts capture trends; 1d Camarilla pivot provides institutional support/resistance levels; volume confirms institutional participation.
Works in bull (breakouts above R4) and bear (breakdowns below S4). Targets 100-200 total trades over 4 years (25-50/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_9291_6h_donchian20_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    typical_price = (high + low + close) / 3
    range_ = high - low
    
    # Camarilla levels
    r4 = close + range_ * CAMARILLA_MULT * 1.5
    r3 = close + range_ * CAMARILLA_MULT * 1.25
    r2 = close + range_ * CAMARILLA_MULT * 1.166
    r1 = close + range_ * CAMARILLA_MULT * 1.083
    
    s1 = close - range_ * CAMARILLA_MULT * 1.083
    s2 = close - range_ * CAMARILLA_MULT * 1.166
    s3 = close - range_ * CAMARILLA_MULT * 1.25
    s4 = close - range_ * CAMARILLA_MULT * 1.5
    
    return r4, r3, r2, r1, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize Camarilla arrays
    r4_1d = np.full(len(close_1d), np.nan)
    r3_1d = np.full(len(close_1d), np.nan)
    s3_1d = np.full(len(close_1d), np.nan)
    s4_1d = np.full(len(close_1d), np.nan)
    
    # Calculate Camarilla for each 1d bar
    for i in range(len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            r4, r3, _, _, _, _, s3, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
            r4_1d[i] = r4
            r3_1d[i] = r3
            s3_1d[i] = s3
            s4_1d[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
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
        if np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]):
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
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Camarilla-based filters
        # Long: breakout above R4 (strong bullish) or breakdown below S3 with reversal (bullish reversal)
        long_breakout_strong = long_breakout and close[i] > r4_1d_aligned[i]
        long_reversal = not long_breakout and close[i] < s3_1d_aligned[i] and close[i] > open[i]  # Bullish candle
        
        # Short: breakdown below S4 (strong bearish) or breakout above R3 with reversal (bearish reversal)
        short_breakout_strong = short_breakout and close[i] < s4_1d_aligned[i]
        short_reversal = not short_breakout and close[i] > r3_1d_aligned[i] and close[i] < open[i]  # Bearish candle
        
        # Entry conditions
        long_entry = (long_breakout_strong or long_reversal) and volume_confirmed
        short_entry = (short_breakout_strong or short_reversal) and volume_confirmed
        
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