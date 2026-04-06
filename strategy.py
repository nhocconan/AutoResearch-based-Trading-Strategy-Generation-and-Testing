#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour strategy using daily Donchian channel breakouts with volume confirmation and ATR stop-loss.
# Goes long when price breaks above the 20-day Donchian high with volume, short when breaks below the 20-day low with volume.
# Uses daily trend (EMA50) as filter to avoid counter-trend trades.
# Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.

name = "exp_13760_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20  # Daily Donchian period
TREND_EMA_PERIOD = 50  # Daily EMA for trend filter
VOLUME_MA_PERIOD = 8   # Volume moving average period
VOLUME_THRESHOLD = 1.5 # Volume must be 1.5x average
SIGNAL_SIZE = 0.25     # Position size (25% of capital)
ATR_PERIOD = 14        # ATR period for stop-loss
ATR_STOP_MULTIPLIER = 2.0 # ATR multiplier for stop-loss

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

def calculate_donchian(high, low, period):
    """Calculate Donchian channel high and low"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data for Donchian channels and trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate daily Donchian channels (using previous day's data to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    upper_1d = np.full_like(high_1d, np.nan)
    lower_1d = np.full_like(low_1d, np.nan)
    
    # Calculate Donchian for each day using previous period's data
    for i in range(DONCHIAN_PERIOD, len(high_1d)):
        upper, lower = calculate_donchian(high_1d[:i], low_1d[:i], DONCHIAN_PERIOD)
        upper_1d[i] = upper[-1]
        lower_1d[i] = lower[-1]
    
    # Align Donchian levels to 4h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # 4h data for entry timing and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_PERIOD, DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or np.isnan(volume_ma[i]):
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
        
        # Trend direction from daily EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Donchian breakout signals
        long_signal = volume_ok and above_ema and close[i] > upper_1d_aligned[i]
        short_signal = volume_ok and below_ema and close[i] < lower_1d_aligned[i]
        
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
            # Exit long on close below Donchian lower (mean reversion)
            if close[i] < lower_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above Donchian upper (mean reversion)
            if close[i] > upper_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals