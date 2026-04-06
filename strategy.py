#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy using 4-hour Donchian(20) breakouts with 1-day EMA(50) trend filter and volume confirmation.
# Uses 1-day trend for direction, 4-hour Donchian breakouts for entries, volume for confirmation.
# Designed for ~100-150 total trades over 4 years (25-38/year) to avoid excessive fees.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.
# Target: 100-200 total trades, 0.20 position size, max DD < -50%.

name = "exp_13734_1h_donchian20_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters - tuned for moderate trade frequency
DONCHIAN_PERIOD = 20
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 8
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 4h and 1d data for filters ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h indicators for entries
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # ATR for stop loss (using 1h data)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # 4h Donchian channels
    donchian_high = pd.Series(high_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for 4h
    volume_ma_4h = pd.Series(volume_4h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ma_4h[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation (using 4h volume)
        volume_ok = volume_4h[i] > (volume_ma_4h[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 1d EMA
        above_ema = close_4h[i] > ema_1d_aligned[i]
        below_ema = close_4h[i] < ema_1d_aligned[i]
        
        # Donchian breakout signals
        if i > 0 and not np.isnan(donchian_high[i-1]) and not np.isnan(donchian_low[i-1]):
            high_prev = donchian_high[i-1]
            low_prev = donchian_low[i-1]
            
            # Long signal: price breaks above Donchian high in uptrend
            long_signal = volume_ok and above_ema and close_4h[i] > high_prev and close_4h[i-1] <= high_prev
            
            # Short signal: price breaks below Donchian low in downtrend
            short_signal = volume_ok and below_ema and close_4h[i] < low_prev and close_4h[i-1] >= low_prev
        else:
            long_signal = False
            short_signal = False
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on opposite Donchian break
            if i > 0 and not np.isnan(donchian_low[i-1]) and not np.isnan(donchian_low[i]):
                low_prev = donchian_low[i-1]
                if close_4h[i] < low_prev and close_4h[i-1] >= low_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = SIGNAL_SIZE
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite Donchian break
            if i > 0 and not np.isnan(donchian_high[i-1]) and not np.isnan(donchian_high[i]):
                high_prev = donchian_high[i-1]
                if close_4h[i] > high_prev and close_4h[i-1] <= high_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals