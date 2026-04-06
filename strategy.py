#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum breakout with 4h trend filter, volume confirmation, and daily volatility filter.
# Uses 4h EMA for trend direction, 1h Donchian breakout for entry, volume surge for confirmation, and daily ATR for volatility filtering.
# Works in bull markets (breakouts above upper band in uptrend) and bear markets (breakdowns below lower band in downtrend).
# Session filter (08-20 UTC) reduces noise. Target: 75-150 total trades over 4 years (19-38/year).

name = "exp_13514_1h_momentum_4h_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_FAST = 9
EMA_SLOW = 21
EMA_4H_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
ATR_PERIOD = 14
ATR_FILTER_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.20

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
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
    if n < 50:
        return np.zeros(n)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, EMA_4H_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (1h)
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # EMA crossover (1h)
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    
    # Volume MA (1h)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR (1h) for stop loss
    atr_1h = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_SLOW, EMA_4H_PERIOD, DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Skip if not in session
        if not in_session:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if indicators not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ma[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_1h[i])):
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
        
        # Trend filter: 4h EMA direction
        uptrend_4h = close[i] > ema_4h_aligned[i]
        downtrend_4h = close[i] < ema_4h_aligned[i]
        
        # Momentum filter: 1h EMA crossover
        bullish_momentum = ema_fast[i] > ema_slow[i]
        bearish_momentum = ema_fast[i] < ema_slow[i]
        
        # Volatility filter: current ATR > 1.5 * daily ATR (avoid low volatility periods)
        vol_filter = atr_1h[i] > (ATR_FILTER_MULTIPLIER * atr_1d_aligned[i])
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout signals using Donchian channels
        breakout_up = high[i] > highest_high[i-1]
        breakout_down = low[i] < lowest_low[i-1]
        
        # Entry conditions
        long_entry = (uptrend_4h and bullish_momentum and vol_filter and volume_ok and breakout_up)
        short_entry = (downtrend_4h and bearish_momentum and vol_filter and volume_ok and breakout_down)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr_1h[i])  # 2*ATR stop loss
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr_1h[i])  # 2*ATR stop loss
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals