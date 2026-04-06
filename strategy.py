#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour strategy using 1d Donchian(20) breakouts with 1w EMA(50) trend filter and volume confirmation.
# Uses 1w trend for direction (bull/bear filter), 1d Donchian breakouts for entries, volume for confirmation.
# Designed for ~100-150 total trades over 4 years (25-38/year) to avoid excessive fees.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.
# Target: 100-200 total trades, 0.25 position size, max DD < -50%.

name = "exp_13741_4h_donchian20_1w_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters - tuned for moderate trade frequency
DONCHIAN_PERIOD = 20
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 8
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
    
    # Load 1d and 1w data for filters ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, TREND_EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d indicators for entries
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # ATR for stop loss (using 4h data)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # 1d Donchian channels
    donchian_high = pd.Series(high_1d).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low_1d).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for 1d
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ma_1d[i]):
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
        
        # Volume confirmation (using 1d volume)
        volume_ok = volume_1d[i] > (volume_ma_1d[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 1w EMA
        above_ema = close_1d[i] > ema_1w_aligned[i]
        below_ema = close_1d[i] < ema_1w_aligned[i]
        
        # Donchian breakout signals
        if i > 0 and not np.isnan(donchian_high[i-1]) and not np.isnan(donchian_low[i-1]):
            high_prev = donchian_high[i-1]
            low_prev = donchian_low[i-1]
            
            # Long signal: price breaks above Donchian high in uptrend
            long_signal = volume_ok and above_ema and close_1d[i] > high_prev and close_1d[i-1] <= high_prev
            
            # Short signal: price breaks below Donchian low in downtrend
            short_signal = volume_ok and below_ema and close_1d[i] < low_prev and close_1d[i-1] >= low_prev
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
                if close_1d[i] < low_prev and close_1d[i-1] >= low_prev:
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
                if close_1d[i] > high_prev and close_1d[i-1] <= high_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals