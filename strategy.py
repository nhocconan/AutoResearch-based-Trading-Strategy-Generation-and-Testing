#!/usr/bin/env python3
"""
Experiment #8014: 1-hour strategy with 4h/1d trend filter and volume confirmation.
Hypothesis: In both bull and bear markets, price breaks of 1-hour ranges with volume confirmation
and aligned 4h/1d trend direction capture meaningful moves. Using higher timeframes for trend
reduces whipsaw, while volume filters ensure momentum. Target: 60-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8014_1h_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RANGE_PERIOD = 20          # For 1h high/low range
VOLUME_MA_PERIOD = 20      # Volume moving average
VOLUME_THRESHOLD = 1.5     # Volume must be 1.5x average
SIGNAL_SIZE = 0.20         # Position size (20% of capital)
ATR_PERIOD = 14            # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.0  # Stop loss at 2x ATR

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h trend: price above/below 20-period EMA
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_4h = np.where(close_4h > ema_4h, 1, -1)  # 1=uptrend, -1=downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d trend: price above/below 34-period EMA
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)  # 1=uptrend, -1=downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h price range (highest high/lowest low over period)
    highest_high = pd.Series(high).rolling(window=RANGE_PERIOD, min_periods=RANGE_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=RANGE_PERIOD, min_periods=RANGE_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
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
    start = max(RANGE_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, 20, 34) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]):
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
        
        # Determine if trends are aligned (both same direction)
        bullish_aligned = (trend_4h_aligned[i] == 1) and (trend_1d_aligned[i] == 1)
        bearish_aligned = (trend_4h_aligned[i] == -1) and (trend_1d_aligned[i] == -1)
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions - price breaks 1h range
        upper_breakout = close[i] > highest_high[i-1] if i-1 >= 0 and not np.isnan(highest_high[i-1]) else False
        lower_breakout = close[i] < lowest_low[i-1] if i-1 >= 0 and not np.isnan(lowest_low[i-1]) else False
        
        # Entry conditions: aligned trend + volume + breakout
        long_entry = bullish_aligned and upper_breakout and volume_confirmed
        short_entry = bearish_aligned and lower_breakout and volume_confirmed
        
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