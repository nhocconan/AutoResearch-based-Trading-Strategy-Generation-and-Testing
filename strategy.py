#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13934_1h_4h_1d_trend_filter_v1"
timeframe = "1h"
leverage = 1.0

# Hypothesis: 1h strategy using 4h trend direction (EMA50) and 1d volatility filter (ATR ratio)
# Enter long when: 1h price > 4h EMA50 AND 1d ATR ratio > 1.5 (high volatility) AND 1h close > 1h open
# Enter short when: 1h price < 4h EMA50 AND 1d ATR ratio > 1.5 AND 1h close < 1h open
# Exit when trend changes or volatility drops
# Uses session filter (08-20 UTC) to avoid low-liquidity hours
# Target: 60-150 trades over 4 years by requiring both trend and volatility alignment
# Works in bull (trend + volatility) and bear (trend + volatility) regimes

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend filter ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, 50)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data for volatility filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # 1h ATR for stop loss
    atr_1h = calculate_atr(high, low, close, 14)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(50, 14) + 1
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Skip if required data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(atr_1h[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: 1d ATR ratio > 1.5 (current vs 20-period average)
        atr_ma_20 = np.mean(atr_1d_aligned[max(0, i-19):i+1]) if i >= 19 else atr_1d_aligned[i]
        vol_filter = atr_1d_aligned[i] > (atr_ma_20 * 1.5)
        
        # 1h price action filter: close > open for long, close < open for short
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        # Trend filter from 4h EMA
        trend_up = close[i] > ema_4h_aligned[i]
        trend_down = close[i] < ema_4h_aligned[i]
        
        # Entry signals
        long_signal = vol_filter and trend_up and bullish_candle
        short_signal = vol_filter and trend_down and bearish_candle
        
        # Check stops
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
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr_1h[i])
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr_1h[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on trend change or volatility drop
            if not trend_up or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short on trend change or volatility drop
            if not trend_down or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals