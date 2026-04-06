#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12574_1h_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters - Target 60-150 total trades over 4 years (15-37/year)
DONCHIAN_PERIOD = 20          # 4h Donchian for trend direction
EMA_PERIOD = 50               # 1d EMA for trend filter
VOLUME_MA_PERIOD = 20         # Volume confirmation
VOLUME_THRESHOLD = 2.0        # Volume spike requirement
SIGNAL_SIZE = 0.20            # Position size (20%)
ATR_PERIOD = 14               # ATR for volatility
ATR_STOP_MULTIPLIER = 2.0     # Stop loss multiplier
SESSION_START_HOUR = 8        # 08:00 UTC
SESSION_END_HOUR = 20         # 20:00 UTC

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Set first value to first true range to avoid NaN
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_donchian(high, low, period):
    """Calculate Donchian channels with proper min_periods"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute session hours ONCE before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 1d EMA for trend
    ema_1d = calculate_ema(df_1d['close'].values, EMA_PERIOD)
    
    # Calculate ATR on 1h data
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    volume_1h = prices['volume'].values
    
    # Calculate indicators
    upper_4h, lower_4h = calculate_donchian(high_4h, low_4h, DONCHIAN_PERIOD)
    volume_ma = pd.Series(volume_1h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high_1h, low_1h, close_1h, ATR_PERIOD)
    
    # Align HTF indicators to LTF with proper shift (avoid look-ahead)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Warmup period - ensure all indicators are valid
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: only trade 08:00-20:00 UTC
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any HTF data not available
        if np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close_1h[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close_1h[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation (require volume spike)
        volume_ok = volume_1h[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filters
        uptrend_4h = close_1h[i] > upper_4h_aligned[i]   # Price above 4h Donchian upper = uptrend
        downtrend_4h = close_1h[i] < lower_4h_aligned[i] # Price below 4h Donchian lower = downtrend
        
        uptrend_1d = close_1h[i] > ema_1d_aligned[i]     # Price above 1d EMA = uptrend
        downtrend_1d = close_1h[i] < ema_1d_aligned[i]   # Price below 1d EMA = downtrend
        
        # Entry conditions - require alignment across timeframes
        long_entry = volume_ok and uptrend_4h and uptrend_1d
        short_entry = volume_ok and downtrend_4h and downtrend_1d
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close_1h[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close_1h[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals