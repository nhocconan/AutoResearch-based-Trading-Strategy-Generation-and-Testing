#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with weekly trend filter and volume confirmation.
# Uses 1d price channel (Donchian) for entry/exit and 1w EMA for trend direction.
# Only takes long when price breaks above Donchian upper band in uptrend,
# only takes short when price breaks below Donchian lower band in downtrend.
# Volume confirmation ensures institutional participation.
# Designed for low trade frequency (~15-25/year) to minimize fee drag.

name = "exp_13610_1d_donchian20_1w_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands"""
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
    
    # Calculate 1d indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel
    upper_band, lower_band = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
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
        if np.isnan(ema_1w_slope_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(volume_ma[i]):
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
        if i > 0 and not np.isnan(upper_band[i-1]) and not np.isnan(lower_band[i-1]):
            upper_prev = upper_band[i-1]
            upper_curr = upper_band[i]
            lower_prev = lower_band[i-1]
            lower_curr = lower_band[i]
            close_prev = close[i-1]
            close_curr = close[i]
            
            # Long signal: price breaks above upper band in uptrend
            long_signal = volume_ok and uptrend and close_prev <= upper_prev and close_curr > upper_curr
            
            # Short signal: price breaks below lower band in downtrend
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
            # Exit long on opposite Donchian break or stop loss
            if i > 0 and not np.isnan(lower_band[i-1]) and not np.isnan(lower_band[i]):
                lower_prev = lower_band[i-1]
                lower_curr = lower_band[i]
                close_prev = close[i-1]
                close_curr = close[i]
                # Exit if price breaks below lower band (loss of uptrend)
                if close_prev >= lower_prev and close_curr < lower_curr:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = SIGNAL_SIZE
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite Donchian break or stop loss
            if i > 0 and not np.isnan(upper_band[i-1]) and not np.isnan(upper_band[i]):
                upper_prev = upper_band[i-1]
                upper_curr = upper_band[i]
                close_prev = close[i-1]
                close_curr = close[i]
                # Exit if price breaks above upper band (loss of downtrend)
                if close_prev <= upper_prev and close_curr > upper_curr:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals