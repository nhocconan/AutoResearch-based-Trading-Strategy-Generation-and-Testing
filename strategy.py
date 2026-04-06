#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with daily EMA trend filter and volume confirmation.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low.
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with price above daily EMA50.
# Short when Bear Power > 0 and rising, Bull Power < 0 and falling, with price below daily EMA50.
# Volume confirmation ensures institutional participation. Works in bull markets (strong Bull Power)
# and bear markets (strong Bear Power). Target: 50-150 total trades over 4 years.

name = "elder_ray_6h_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ER_EMA_PERIOD = 13    # Elder Ray EMA period
DAILY_EMA_PERIOD = 50 # Daily trend filter
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_vals = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    daily_ema = ema(close_1d, DAILY_EMA_PERIOD)
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components: EMA(13) of close
    er_ema = ema(close, ER_EMA_PERIOD)
    bull_power = high - er_ema
    bear_power = er_ema - low
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr_vals = atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ER_EMA_PERIOD, DAILY_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(daily_ema_aligned[i]) or np.isnan(er_ema[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Elder Ray signals with slope confirmation
        bull_rising = bull_power[i] > bull_power[i-1]
        bull_falling = bull_power[i] < bull_power[i-1]
        bear_rising = bear_power[i] > bear_power[i-1]
        bear_falling = bear_power[i] < bear_power[i-1]
        
        # Trend filter: price relative to daily EMA
        price_above_ema = close[i] > daily_ema_aligned[i]
        price_below_ema = close[i] < daily_ema_aligned[i]
        
        # Long: Bull Power positive and rising, price above daily EMA
        long_signal = volume_ok and bull_power[i] > 0 and bull_rising and price_above_ema
        # Short: Bear Power positive and rising, price below daily EMA
        short_signal = volume_ok and bear_power[i] > 0 and bear_rising and price_below_ema
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_vals[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_vals[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals