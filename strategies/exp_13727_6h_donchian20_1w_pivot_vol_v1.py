#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA(50) trend filter and volume confirmation (1.5x volume).
# Uses 1d trend for direction, 4h Donchian breakouts for entries, volume for confirmation.
# Designed for ~100 total trades over 4 years (25/year) to avoid excessive fees.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.
# Target: 75-200 total trades, 0.25 position size, max DD < -50%.

name = "exp_13726_4h_donchian20_1d_ema_vol_v1"
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
    
    # Load 1d data for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ma[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Donchian breakout signals
        if i > 0 and not np.isnan(donchian_high[i-1]) and not np.isnan(donchian_low[i-1]):
            high_prev = donchian_high[i-1]
            low_prev = donchian_low[i-1]
            
            # Long signal: price breaks above Donchian high in uptrend
            long_signal = volume_ok and above_ema and close[i] > high_prev and close[i-1] <= high_prev
            
            # Short signal: price breaks below Donchian low in downtrend
            short_signal = volume_ok and below_ema and close[i] < low_prev and close[i-1] >= low_prev
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
                if close[i] < low_prev and close[i-1] >= low_prev:
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
                if close[i] > high_prev and close[i-1] <= high_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals