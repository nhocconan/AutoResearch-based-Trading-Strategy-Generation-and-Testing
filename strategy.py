#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12774_1h_4h_1d_trend_filter"
timeframe = "1h"
leverage = 1.0

# Parameters
HTF1_TREND_PERIOD = 21   # 4h EMA period for trend
HTF2_TREND_PERIOD = 50   # 1d EMA period for filter
ENTRY_CHANNEL_PERIOD = 20  # 1h Donchian for entry timing
VOLUME_MA_PERIOD = 20      # Volume confirmation
VOLUME_THRESHOLD = 1.5     # Volume spike multiplier
SIGNAL_SIZE = 0.20         # Position size (20% of capital)
ATR_PERIOD = 14            # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.5  # Stop loss multiplier

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
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA for trend
    ema_4h = pd.Series(close_4h).ewm(span=HTF1_TREND_PERIOD, adjust=False, min_periods=HTF1_TREND_PERIOD).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA for filter
    ema_1d = pd.Series(close_1d).ewm(span=HTF2_TREND_PERIOD, adjust=False, min_periods=HTF2_TREND_PERIOD).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # 1h Donchian channels for entry timing
    high_1h = pd.Series(high).rolling(window=ENTRY_CHANNEL_PERIOD, min_periods=ENTRY_CHANNEL_PERIOD).max().values
    low_1h = pd.Series(low).rolling(window=ENTRY_CHANNEL_PERIOD, min_periods=ENTRY_CHANNEL_PERIOD).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(HTF1_TREND_PERIOD, HTF2_TREND_PERIOD, ENTRY_CHANNEL_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if data not ready
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(high_1h[i]) or np.isnan(low_1h[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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
        
        # Determine trend from higher timeframes
        # Long trend: 4h price > 4h EMA AND 1d price > 1d EMA
        long_trend = close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]
        # Short trend: 4h price < 4h EMA AND 1d price < 1d EMA
        short_trend = close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Entry signals based on 1h Donchian breakout in direction of trend
        breakout_long = volume_ok and long_trend and close[i] >= high_1h[i]
        breakout_short = volume_ok and short_trend and close[i] <= low_1h[i]
        
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