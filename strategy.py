#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian channel breakout with 4h trend filter and volume confirmation.
# In trending markets, price tends to continue in the direction of the breakout.
# Use 4h EMA(50) slope to determine trend: only take long when 4h EMA sloping up, short when sloping down.
# Volume confirmation ensures breakout is supported by participation.
# Works in both bull and bear markets by following the 4h trend direction.
# Target: 60-150 total trades over 4 years (15-37/year) with tight entry conditions.

name = "exp_13594_1h_donchian20_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper and lower bands"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    return highest_high.values, lowest_low.values

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
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend filter ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, TREND_EMA_PERIOD)
    ema_4h_slope = np.diff(ema_4h, prepend=ema_4h[0])  # slope approximation
    ema_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_slope)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
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
        if np.isnan(ema_4h_slope_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(volume_ma[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 4h EMA slope
        uptrend = ema_4h_slope_aligned[i] > 0
        downtrend = ema_4h_slope_aligned[i] < 0
        
        # Donchian breakout signals
        # Avoid lookback by checking current and previous values
        if i > 0 and not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
            high_prev = high[i-1]
            low_prev = low[i-1]
            upper_prev = donchian_upper[i-1]
            lower_prev = donchian_lower[i-1]
            
            # Long signal: price breaks above upper Donchian band in uptrend
            long_signal = volume_ok and uptrend and high_prev <= upper_prev and close[i] > upper_prev
            
            # Short signal: price breaks below lower Donchian band in downtrend
            short_signal = volume_ok and downtrend and low_prev >= lower_prev and close[i] < lower_prev
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
            # Exit long on opposite Donchian breakout or stop loss
            if i > 0 and not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
                high_prev = high[i-1]
                low_prev = low[i-1]
                upper_prev = donchian_upper[i-1]
                lower_prev = donchian_lower[i-1]
                # Exit if price breaks below lower Donchian band (trend reversal)
                if low_prev >= lower_prev and close[i] < lower_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = SIGNAL_SIZE
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite Donchian breakout or stop loss
            if i > 0 and not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
                high_prev = high[i-1]
                low_prev = low[i-1]
                upper_prev = donchian_upper[i-1]
                lower_prev = donchian_lower[i-1]
                # Exit if price breaks above upper Donchian band (trend reversal)
                if high_prev <= upper_prev and close[i] > upper_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals