#!/usr/bin/env python3
"""
Experiment #9974: 1h Donchian Breakout + 4h/1d Trend + Volume Spike + Session Filter
Hypothesis: Donchian(20) breakouts on 1h in the direction of 4h EMA50 and 1d EMA200 trend, confirmed by volume spikes and active session (08-20 UTC), provide high-probability trend continuation trades. The 4h/1d trend filters reduce false breakouts, volume confirms institutional interest, and session filter avoids low-liquidity periods. This should work in bull markets (breakouts above trend) and bear markets (breakdowns below trend) by following the higher timeframe trend.
Target: 60-150 total trades over 4 years (15-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9974_1h_donchian_breakout_4h_1d_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.8
EMA_4H_PERIOD = 50
EMA_1D_PERIOD = 200
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load 4h and 1d data ONCE before loop for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, EMA_4H_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA for trend direction
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_1D_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_4H_PERIOD, EMA_1D_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if trend filters not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
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
        
        # Trend filter: price above/both EMAs for long, below/both for short
        above_both_emas = close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]
        below_both_emas = close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of trend with volume and session
        long_entry = bullish_breakout and above_both_emas and volume_spike and in_session
        short_entry = bearish_breakout and below_both_emas and volume_spike and in_session
        
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
</truncated>