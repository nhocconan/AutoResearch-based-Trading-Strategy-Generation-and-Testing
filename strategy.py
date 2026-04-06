#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour 4-hour Donchian(20) breakout with 1-day EMA(200) trend filter, volume confirmation, and session filter (08-20 UTC).
# Uses 1-day trend for direction, 4h for structure, and 1h for precise entry timing.
# Volume confirmation ensures institutional participation. Session filter reduces noise.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Works in bull markets (breakouts with volume in uptrend) and bear markets (breakdowns with volume in downtrend).

name = "exp_13694_1h_4h_donchian20_1d_ema_vol_session_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_EMA_PERIOD = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Donchian channels ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Load 1d data for EMA trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Donchian channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_4h = pd.Series(high_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align HTF data to 1h
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available or outside session
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma[i]) or not in_session[i]):
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
        
        # Donchian breakout signals (using 4h structure)
        if i > 0 and not np.isnan(donchian_high_4h_aligned[i-1]) and not np.isnan(donchian_low_4h_aligned[i-1]):
            high_prev = donchian_high_4h_aligned[i-1]
            low_prev = donchian_low_4h_aligned[i-1]
            
            # Long signal: price breaks above 4h Donchian high in uptrend
            long_signal = volume_ok and above_ema and close[i] > high_prev and close[i-1] <= high_prev
            
            # Short signal: price breaks below 4h Donchian low in downtrend
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
            # Exit long on opposite 4h Donchian break
            if i > 0 and not np.isnan(donchian_low_4h_aligned[i-1]) and not np.isnan(donchian_low_4h_aligned[i]):
                low_prev = donchian_low_4h_aligned[i-1]
                if close[i] < low_prev and close[i-1] >= low_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = SIGNAL_SIZE
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite 4h Donchian break
            if i > 0 and not np.isnan(donchian_high_4h_aligned[i-1]) and not np.isnan(donchian_high_4h_aligned[i]):
                high_prev = donchian_high_4h_aligned[i-1]
                if close[i] > high_prev and close[i-1] <= high_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals