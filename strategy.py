#!/usr/bin/env python3
"""
Experiment #9667: 6h Donchian Breakout + Weekly Trend + Volume Confirmation.
Hypothesis: Donchian(20) breakouts on 6h timeframe, filtered by weekly trend direction 
and volume confirmation, provide high-probability entries. In bull markets (weekly 
trend up), we take long breakouts; in bear markets (weekly trend down), we take 
short breakouts. This adapts to market regime while maintaining clear entry/exit rules.
Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9667_6h_donchian_breakout_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.8
WEEKLY_TREND_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_sma(arr, period):
    """Calculate simple moving average"""
    return pd.Series(arr).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for trend filter)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly trend (SMA 50)
    weekly_close = df_weekly['close'].values
    weekly_sma = calculate_sma(weekly_close, WEEKLY_TREND_PERIOD)
    weekly_sma_aligned = align_htf_to_ltf(prices, df_weekly, weekly_sma)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # ATR for volatility and stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, WEEKLY_TREND_PERIOD, ATR_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if weekly trend data not available
        if np.isnan(weekly_sma_aligned[i]):
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
        
        # Weekly trend direction
        weekly_trend_up = weekly_sma_aligned[i] > weekly_close[0]  # Using first value as baseline, better: compare to previous
        # Actually, let's use slope: current vs previous weekly SMA
        if i >= 1 and not np.isnan(weekly_sma_aligned[i-1]):
            weekly_trend_up = weekly_sma_aligned[i] > weekly_sma_aligned[i-1]
        else:
            weekly_trend_up = True  # default to allow trading until we have data
        
        # Donchian breakout signals
        # Long: price breaks above upper band in uptrend
        # Short: price breaks below lower band in downtrend
        long_breakout = close[i] > donch_upper[i] and weekly_trend_up and volume_spike
        short_breakout = close[i] < donch_lower[i] and not weekly_trend_up and volume_spike
        
        # Entry conditions
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_breakout:
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