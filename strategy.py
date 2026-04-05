#!/usr/bin/env python3
"""
Experiment #8647: 6h Donchian breakout + 1d pivot direction + volume confirmation
Hypothesis: Combining 6h price breakouts with daily pivot point bias and volume confirmation
creates high-probability entries with controlled frequency. Pivot points act as institutional
reference levels (R3/S3 for reversals, R4/S4 for breakouts). Volume confirms institutional
participation. Works in bull/bear by using pivot-based bias rather than simple trend.
Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8647_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
PIVOT_LOOKBACK = 1  # Use previous day's pivot
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot_points(high, low, close):
    """
    Calculate standard pivot points:
    P = (H + L + C) / 3
    R1 = 2*P - L
    S1 = 2*P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    R3 = H + 2*(P - L)
    S3 = L - 2*(H - P)
    R4 = R3 + (H - L)
    S4 = S3 - (H - L)
    """
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
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d pivot points from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels
    P, R1, R2, R3, R4, S1, S2, S3, S4 = calculate_pivot_points(high_1d, low_1d, close_1d)
    
    # Determine bias based on price relative to pivot
    # Above P = bullish bias, Below P = bearish bias
    price_vs_pivot = np.where(close_1d > P, 1, 
                       np.where(close_1d < P, -1, 0))  # 1=bullish, -1=bearish, 0=at pivot
    
    # Strong bias when price is beyond S3/R3 (strong reversal potential)
    strong_bull = close_1d < S3  # Price below S3 = potential bullish reversal
    strong_bear = close_1d > R3  # Price above R3 = potential bearish reversal
    
    # Breakout bias when price breaks S4/R4 (continuation)
    breakout_bull = close_1d > R4  # Break above R4 = bullish continuation
    breakout_bear = close_1d < S4  # Break below S4 = bearish continuation
    
    # Combine: bias = 1 (bullish), -1 (bearish), 0 (neutral)
    bias = np.where(strong_bull | breakout_bull, 1,
            np.where(strong_bear | breakout_bear, -1, 0))
    bias = np.where(price_vs_pivot == 1, np.maximum(bias, 1), bias)  # Reinforce if above pivot
    bias = np.where(price_vs_pivot == -1, np.minimum(bias, -1), bias)  # Reinforce if below pivot
    
    bias_aligned = align_htf_to_ltf(prices, df_1d, bias)
    
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
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, PIVOT_LOOKBACK) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(bias_aligned[i]):
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
        
        # Determine market bias from 1d pivot
        bull_bias = bias_aligned[i] >= 1   # Bullish bias from pivot analysis
        bear_bias = bias_aligned[i] <= -1  # Bearish bias from pivot analysis
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation - require strong volume
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_bias and long_breakout and volume_confirmed
        short_entry = bear_bias and short_breakout and volume_confirmed
        
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