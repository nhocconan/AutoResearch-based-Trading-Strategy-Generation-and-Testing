#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour VWAP mean reversion with weekly trend filter and volatility filter.
# In ranging markets (common in 2025), price reverts to VWAP with high probability.
# Weekly EMA ensures we only trade in direction of higher timeframe trend.
# Volume spike confirms institutional participation at mean reversion points.
# Target: 50-150 total trades over 4 years by using tight entry conditions.

name = "exp_13368_12h_vwap_mean_reversion_trend_vol"
timeframe = "12h"
leverage = 1.0

# Parameters
VWAP_PERIOD = 24  # 2 periods for VWAP (24h = 2*12h)
EMA_WEEKLY = 20   # Weekly EMA
VOLUME_MA = 20    # Volume moving average
VOLUME_THRESHOLD = 1.8  # High volume threshold
DEV_THRESHOLD = 0.015   # 1.5% deviation from VWAP for entry
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_vwap(high, low, close, volume, period):
    """Calculate VWAP"""
    typical_price = (high + low + close) / 3
    vwap_num = np.convolve(typical_price * volume, np.ones(period), 'full')[:len(typical_price)]
    vwap_den = np.convolve(volume, np.ones(period), 'full')[:len(typical_price)]
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(typical_price, np.nan), where=vwap_den!=0)
    return vwap

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_WEEKLY)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP
    vwap = calculate_vwap(high, low, close, volume, VWAP_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA, min_periods=VOLUME_MA).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_WEEKLY, VWAP_PERIOD, VOLUME_MA, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(vwap[i]) or np.isnan(volume_ma[i]):
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
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Deviation from VWAP
        if not np.isnan(vwap[i]) and vwap[i] != 0:
            dev_pct = (close[i] - vwap[i]) / vwap[i]
        else:
            dev_pct = 0
        
        # Mean reversion signals
        oversold = dev_pct < -DEV_THRESHOLD  # Price significantly below VWAP
        overbought = dev_pct > DEV_THRESHOLD  # Price significantly above VWAP
        
        # Generate signals
        if position == 0:
            if oversold and volume_ok and uptrend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif overbought and volume_ok and downtrend:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals