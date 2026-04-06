# 6H 4-WEEK CHANNEL BREAKOUT WITH VOLUME FILTER
# Hypothesis: Price breaking above 4-week high/low with volume confirmation captures strong trends.
# Works in bull/bear because breakouts capture strong moves, volume filters weak signals.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.
# Uses 4-week high/low as structure and volume for confirmation.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13015_6h_4week_channel_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CHANNEL_PERIOD = 20  # 20 * 6h = 5 days
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for 4-week channel
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate 4-week (20-day) high/low
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    
    high_series = pd.Series(high_d)
    low_series = pd.Series(low_d)
    
    channel_high = high_series.rolling(window=CHANNEL_PERIOD, min_periods=CHANNEL_PERIOD).max().values
    channel_low = low_series.rolling(window=CHANNEL_PERIOD, min_periods=CHANNEL_PERIOD).min().values
    
    # Align to 6h timeframe
    channel_high_aligned = align_htf_to_ltf(prices, df_daily, channel_high)
    channel_low_aligned = align_htf_to_ltf(prices, df_daily, channel_low)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CHANNEL_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if channel levels not available
        if np.isnan(channel_high_aligned[i]) or np.isnan(channel_low_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout above channel high or below channel low
        breakout_long = volume_ok and close[i] >= channel_high_aligned[i]
        breakout_short = volume_ok and close[i] <= channel_low_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
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