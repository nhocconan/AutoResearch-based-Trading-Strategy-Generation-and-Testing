#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and 12h trend filter.
# Long: price breaks above Donchian(20) high + volume > 1.5x average + 12h EMA(50) rising
# Short: price breaks below Donchian(20) low + volume > 1.5x average + 12h EMA(50) falling
# Exit: opposite signal or stop loss at 2.5*ATR. Uses tight entries to limit trades.
# Works in bull markets (captures breakouts) and bear markets (captures breakdowns).

name = "exp_13583_4h_donchian20_1d_vol_12h_trend_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
TREND_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    ema_12h_slope = np.diff(ema_12h, prepend=ema_12h[0])  # slope approximation
    ema_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_slope)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, TREND_EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_12h_slope_aligned[i]):
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
        
        # Volume confirmation (using 1d average)
        volume_ok = volume[i] > (vol_ma_1d_aligned[i] * VOLUME_THRESHOLD)
        
        # Trend filter from 12h EMA slope
        bull_trend = ema_12h_slope_aligned[i] > 0
        bear_trend = ema_12h_slope_aligned[i] < 0
        
        # Breakout signals
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        
        # Generate signals
        if position == 0:
            if volume_ok and bull_trend and long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and bear_trend and short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on short breakout or trend change
            if short_breakout or not bull_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on long breakout or trend change
            if long_breakout or not bear_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals