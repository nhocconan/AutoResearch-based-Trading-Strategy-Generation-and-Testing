#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h EMA trend filter and 1d ADX trend strength filter.
# Uses 4h EMA for trend direction and 1d ADX to ensure we only trade in strong trends.
# This reduces false signals in choppy markets and improves win rate.
# Entry occurs when 1h price crosses 4h EMA with volume confirmation.
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost.

name = "exp_13114_1h_ema4x_adx1d_vol"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_4H_PERIOD = 20
ADX_1D_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+, DM-
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, EMA_4H_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, ADX_1D_PERIOD)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_4H_PERIOD, ADX_1D_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA or ADX not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]):
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
        
        # Trend filters
        uptrend = ema_4h_aligned[i] > 0 and close[i] > ema_4h_aligned[i]
        downtrend = ema_4h_aligned[i] > 0 and close[i] < ema_4h_aligned[i]
        strong_trend = adx_1d_aligned[i] > ADX_THRESHOLD
        
        # Entry signals: price crosses 4h EMA with volume and strong trend
        if position == 0:
            # Long: price crosses above 4h EMA
            if i > 0 and close[i-1] <= ema_4h_aligned[i-1] and close[i] > ema_4h_aligned[i] and volume_ok and strong_trend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: price crosses below 4h EMA
            elif i > 0 and close[i-1] >= ema_4h_aligned[i-1] and close[i] < ema_4h_aligned[i] and volume_ok and strong_trend:
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