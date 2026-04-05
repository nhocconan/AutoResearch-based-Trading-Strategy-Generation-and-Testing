#!/usr/bin/env python3
"""
Experiment #9467: 6h Camarilla Pivot + Volume Spike + Regime Filter.
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
provide high-probability reversal/continuation signals when combined with volume spikes 
and regime filtering (ADX < 25 for mean reversion, ADX > 25 for breakout). 
Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
Works in bull (R4 breakouts) and bear (S4 breakdowns) with mean reversion in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9467_6h_camarilla_pivot_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC for pivot calculation
VOLUME_SPIKE_MULTIPLIER = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return tr

def calculate_adx(high, low, close, period):
    """Calculate ADX using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla_pivot(high, low, close):
    """
    Calculate Camarilla pivot levels
    R4 = close + (high - low) * 1.1/2
    R3 = close + (high - low) * 1.1/4
    R2 = close + (high - low) * 1.1/6
    R1 = close + (high - low) * 1.1/12
    PP = (high + low + close) / 3
    S1 = close - (high - low) * 1.1/12
    S2 = close - (high - low) * 1.1/6
    S3 = close - (high - low) * 1.1/4
    S4 = close - (high - low) * 1.1/2
    """
    range_hl = high - low
    r4 = close + range_hl * 1.1 / 2
    r3 = close + range_hl * 1.1 / 4
    s3 = close - range_hl * 1.1 / 4
    s4 = close - range_hl * 1.1 / 2
    return r3, r2, r1, pp, s1, s2, s3, s4  # We'll use R3, S3 for mean reversion and R4, S4 for breakout

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivot calculation)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels - we need R3, S3 for mean reversion and R4, S4 for breakout
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.1 / 4
    s3_1d = close_1d - range_1d * 1.1 / 4
    r4_1d = close_1d + range_1d * 1.1 / 2
    s4_1d = close_1d - range_1d * 1.1 / 2
    
    # Align 1d levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX for regime filtering
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, ADX_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Regime filter: ADX < 25 for mean reversion, ADX > 25 for breakout
        low_volatility = adx[i] < ADX_THRESHOLD   # Ranging market
        high_volatility = adx[i] >= ADX_THRESHOLD  # Trending market
        
        # Mean reversion signals (ADX < 25): fade at R3/S3
        mean_rev_long = low_volatility and volume_spike and close[i] <= s3_1d_aligned[i]
        mean_rev_short = low_volatility and volume_spike and close[i] >= r3_1d_aligned[i]
        
        # Breakout signals (ADX >= 25): break at R4/S4
        breakout_long = high_volatility and volume_spike and close[i] >= r4_1d_aligned[i]
        breakout_short = high_volatility and volume_spike and close[i] <= s4_1d_aligned[i]
        
        # Entry conditions
        long_entry = mean_rev_long or breakout_long
        short_entry = mean_rev_short or breakout_short
        
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