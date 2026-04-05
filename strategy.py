#!/usr/bin/env python3
"""
Experiment #8971: 6h Donchian breakout + 1d pivot reversal + volume confirmation.
Hypothesis: Combines trend-following (Donchian breakout) with mean-reversion (pivot rejection) 
using 1d pivot levels for context. In trending markets, breakouts at R4/S4 continue; 
in ranging markets, reversals at R3/S3 capture mean reversion. Volume filters ensure 
institutional participation. Works in bull (breakouts) and bear (reversals at resistance).
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_8971_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
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

def calculate_pivot_levels(high, low, close):
    """Calculate classic pivot points: P = (H+L+C)/3, then support/resistance levels"""
    P = (high + low + close) / 3.0
    R1 = 2*P - low
    S1 = 2*P - high
    R2 = P + (high - low)
    S2 = P - (high - low)
    R3 = high + 2*(P - low)
    S3 = low - 2*(high - P)
    R4 = R3 + (high - low)
    S4 = S3 - (high - low)
    return P, R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use only completed daily bars (no look-ahead)
    P, R1, R2, R3, R4, S1, S2, S3, S4 = calculate_pivot_levels(high_1d, low_1d, close_1d)
    P = np.roll(P, 1)
    R1 = np.roll(R1, 1)
    R2 = np.roll(R2, 1)
    R3 = np.roll(R3, 1)
    R4 = np.roll(R4, 1)
    S1 = np.roll(S1, 1)
    S2 = np.roll(S2, 1)
    S3 = np.roll(S3, 1)
    S4 = np.roll(S4, 1)
    
    # Align pivot levels to 6h timeframe
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
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
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(P_aligned[i]):
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
        
        # Determine market context from pivot levels
        # Strong resistance/support levels (R4/S4) - breakout continuation
        strong_resistance = R4_aligned[i]
        strong_support = S4_aligned[i]
        
        # Mean reversion levels (R3/S3) - rejection/reversal
        resistance = R3_aligned[i]
        support = S3_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Pivot-based conditions
        # Breakout at strong levels (R4/S4) - continuation
        breakout_at_resistance = long_breakout and close[i] > strong_resistance
        breakout_at_support = short_breakout and close[i] < strong_support
        
        # Reversal at mean reversion levels (R3/S3) - rejection
        rejection_at_resistance = (close[i] > resistance and 
                                  close[i] < resistance + 0.1 * atr[i] and  # Near resistance
                                  high[i] > resistance)                   # Tested resistance
        rejection_at_support = (close[i] < support and 
                               close[i] > support - 0.1 * atr[i] and     # Near support
                               low[i] < support)                         # Tested support
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        # Long: breakout continuation at resistance OR rejection at support
        long_entry = ((breakout_at_resistance or rejection_at_support) and volume_confirmed)
        # Short: breakout continuation at support OR rejection at resistance
        short_entry = ((breakout_at_support or rejection_at_resistance) and volume_confirmed)
        
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