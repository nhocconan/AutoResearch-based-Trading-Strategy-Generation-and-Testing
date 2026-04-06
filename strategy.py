#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index with 1w/1d regime filter
# Uses weekly and daily Choppiness Index to determine market regime
# When weekly CHOP > 61.8 (ranging) and daily CHOP < 38.2 (trending): breakout strategy
# When weekly CHOP < 38.2 (trending) and daily CHOP > 61.8 (ranging): mean reversion
# Otherwise: no trade
# Choppiness Index formula: CHOP = 100 * log10(SUM(ATR, n) / (MAX(HIGH, n) - MIN(LOW, n))) / log10(n)
# Target: 50-150 total trades over 4 years with regime-adaptive logic
# Uses weekly regime to avoid trading against higher timeframe trend

name = "6h_chop_regime_1w_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Sum of True Range over period
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / (max_high - min_low + 1e-10)) / np.log10(period)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1w data for regime filter (weekly chop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    chop_1w = calculate_chop(high_1w, low_1w, close_1w, 14)
    
    # 1d data for regime filter and signals (daily chop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Align higher timeframe data to 6h timeframe
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(chop_1w_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        chop_w = chop_1w_aligned[i]
        chop_d = chop_1d_aligned[i]
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif chop_w > 61.8 and chop_d > 61.8:  # Both ranging - exit
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            elif chop_w < 38.2 and chop_d < 38.2:  # Both trending - consider exit if overextended
                # Simple exit: price far from recent average
                recent_avg = np.mean(close[max(0, i-10):i+1])
                if abs(close[i] - recent_avg) / recent_avg > 0.05:  # 5% deviation
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit conditions based on regime
            elif chop_w > 61.8 and chop_d > 61.8:  # Both ranging - exit
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            elif chop_w < 38.2 and chop_d < 38.2:  # Both trending - consider exit if overextended
                recent_avg = np.mean(close[max(0, i-10):i+1])
                if abs(close[i] - recent_avg) / recent_avg > 0.05:  # 5% deviation
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on regime combinations
            # Case 1: Weekly ranging (61.8) + Daily trending (<38.2) -> Breakout
            if chop_w > 61.8 and chop_d < 38.2:
                # Breakout: price breaks recent high/low
                recent_high = np.max(high[max(0, i-5):i+1])
                recent_low = np.min(low[max(0, i-5):i+1])
                if close[i] > recent_high:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < recent_low:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            # Case 2: Weekly trending (<38.2) + Daily ranging (>61.8) -> Mean reversion
            elif chop_w < 38.2 and chop_d > 61.8:
                # Mean reversion: fade extremes
                recent_avg = np.mean(close[max(0, i-10):i+1])
                recent_std = np.std(close[max(0, i-10):i+1])
                if recent_std > 0:
                    z_score = (close[i] - recent_avg) / recent_std
                    if z_score < -1.5:  # Oversold - long
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    elif z_score > 1.5:  # Overbought - short
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
    
    return signals