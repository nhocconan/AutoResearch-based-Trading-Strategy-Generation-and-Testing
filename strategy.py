#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
# Donchian breakouts capture momentum in trending markets. The 1d EMA filter ensures we only trade in the direction
# of the higher timeframe trend, avoiding counter-trend breakouts in ranging markets. Volume confirmation ensures
# institutional participation. This strategy works in both bull and bear markets by following the trend direction.
# Target: 12-37 trades/year by requiring Donchian breakout + trend alignment + volume spike.

name = "exp_13645_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max()
    lower = pd.Series(low).rolling(window=period, min_periods=period).min()
    return upper.values, lower.values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    ema_1d_slope = np.diff(ema_1d, prepend=ema_1d[0])  # slope approximation
    ema_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slope)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    upper, lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
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
        if np.isnan(ema_1d_slope_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(volume_ma[i]):
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
        
        # Trend direction from 1d EMA slope
        uptrend = ema_1d_slope_aligned[i] > 0
        downtrend = ema_1d_slope_aligned[i] < 0
        
        # Donchian breakout signals
        # Avoid lookback by checking current and previous values
        if i > 0 and not np.isnan(upper[i-1]) and not np.isnan(lower[i-1]):
            upper_prev = upper[i-1]
            lower_prev = lower[i-1]
            upper_curr = upper[i]
            lower_curr = lower[i]
            close_prev = close[i-1]
            close_curr = close[i]
            
            # Long signal: break above upper Donchian in uptrend with volume
            long_signal = volume_ok and uptrend and close_prev <= upper_prev and close_curr > upper_curr
            
            # Short signal: break below lower Donchian in downtrend with volume
            short_signal = volume_ok and downtrend and close_prev >= lower_prev and close_curr < lower_curr
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
            if i > 0 and not np.isnan(upper[i-1]) and not np.isnan(lower[i-1]):
                upper_prev = upper[i-1]
                lower_prev = lower[i-1]
                upper_curr = upper[i]
                lower_curr = lower[i]
                close_prev = close[i-1]
                close_curr = close[i]
                # Exit if price breaks below lower Donchian (end of uptrend)
                if close_prev > lower_prev and close_curr < lower_curr:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = SIGNAL_SIZE
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite Donchian breakout or stop loss
            if i > 0 and not np.isnan(upper[i-1]) and not np.isnan(lower[i-1]):
                upper_prev = upper[i-1]
                lower_prev = lower[i-1]
                upper_curr = upper[i]
                lower_curr = lower[i]
                close_prev = close[i-1]
                close_curr = close[i]
                # Exit if price breaks above upper Donchian (end of downtrend)
                if close_prev < upper_prev and close_curr > upper_curr:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals