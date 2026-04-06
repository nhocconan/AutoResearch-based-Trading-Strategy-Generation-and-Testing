#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w trend filter and volume confirmation.
# Donchian channel breakouts capture strong momentum moves. Use 1w EMA(20) for trend filter: 
# only take breakouts in trend direction. Volume confirms institutional participation.
# ATR-based stop loss manages risk. Target: 12-37 trades/year by using strict breakout 
# conditions (new 20-period high/low) + trend + volume.

name = "exp_13628_12h_donchian20_1w_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands"""
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
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, TREND_EMA_PERIOD)
    ema_1w_slope = np.diff(ema_1w, prepend=ema_1w[0])  # slope approximation
    ema_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_slope)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    upper, lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
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
        if np.isnan(ema_1w_slope_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(volume_ma[i]):
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
        
        # Trend direction from 1w EMA slope
        uptrend = ema_1w_slope_aligned[i] > 0
        downtrend = ema_1w_slope_aligned[i] < 0
        
        # Donchian breakout signals
        # Avoid lookback by checking current and previous values
        if i > 0:
            high_prev = high[i-1]
            low_prev = low[i-1]
            upper_prev = upper[i-1]
            lower_prev = lower[i-1]
            
            # Long signal: price breaks above upper band in uptrend
            long_signal = volume_ok and uptrend and high[i] > upper_prev and high_prev <= upper_prev
            
            # Short signal: price breaks below lower band in downtrend
            short_signal = volume_ok and downtrend and low[i] < lower_prev and low_prev >= lower_prev
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
            # Exit long on opposite Donchian break or stop loss
            if i > 0:
                lower_prev = lower[i-1]
                # Exit if price breaks below lower band (trend reversal)
                if low[i] < lower_prev and low_prev >= lower_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = SIGNAL_SIZE
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite Donchian break or stop loss
            if i > 0:
                upper_prev = upper[i-1]
                # Exit if price breaks above upper band (trend reversal)
                if high[i] > upper_prev and high_prev <= upper_prev:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals