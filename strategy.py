#!/usr/bin/env python3
"""
Experiment #8367: 6-hour weekly pivot mean reversion with daily trend filter.
Hypothesis: Price reverting from extreme weekly pivot levels (S4/R4) during counter-trend
daily conditions captures mean reversion moves in ranging markets while avoiding
strong trends. Weekly pivots provide institutional support/resistance, and daily
counter-trend filters prevent fighting strong momentum. Targets 75-200 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8367_6h_weekly_pivot_meanrev_dailyfilter_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 5  # weeks for pivot calculation
PIVOT_THRESHOLD = 0.02  # 2% from S4/R4 to trigger
RSI_PERIOD = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivots(high, low, close):
    """Calculate weekly pivot points (S1-S4, R1-R4)"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivots from prior week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Initialize pivot arrays
    r4 = np.full_like(close_1w, np.nan)
    s4 = np.full_like(close_1w, np.nan)
    
    # Calculate pivots for each week (using prior week's data)
    for i in range(1, len(close_1w)):
        _, _, _, _, r4[i], _, _, _, s4[i] = calculate_pivots(
            high_1w[i-1], low_1w[i-1], close_1w[i-1]
        )
    
    # Align weekly pivots to 6h
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Calculate daily RSI for trend filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, RSI_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
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
        
        # Calculate distance from pivot levels as percentage
        dist_to_r4 = (r4_aligned[i] - close[i]) / close[i] if r4_aligned[i] > 0 else 0
        dist_to_s4 = (close[i] - s4_aligned[i]) / close[i] if s4_aligned[i] > 0 else 0
        
        # Extreme conditions: near S4/R4 (within threshold)
        near_s4 = dist_to_s4 <= PIVOT_THRESHOLD and s4_aligned[i] > 0
        near_r4 = dist_to_r4 <= PIVOT_THRESHOLD and r4_aligned[i] > 0
        
        # Daily RSI conditions for counter-trend
        rsi_overbought = rsi_1d_aligned[i] >= RSI_OVERBOUGHT
        rsi_oversold = rsi_1d_aligned[i] <= RSI_OVERSOLD
        
        # Entry conditions
        long_entry = near_s4 and rsi_oversold  # Near support with oversold daily RSI
        short_entry = near_r4 and rsi_overbought  # Near resistance with overbought daily RSI
        
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