#!/usr/bin/env python3
"""
Experiment #10147: 6h Elder Ray + ADX + 1d Trend Filter
Hypothesis: Elder Ray power measures bull/bear strength; combined with ADX for trend strength and 1d EMA filter, 
it identifies strong directional moves. Works in bull markets (strong bull power + ADX > 20) and bear markets 
(strong bear power + ADX > 20). Designed for low trade frequency (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10147_6h_elder_ray_adx_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_PERIOD = 13
ADX_PERIOD = 14
ADX_THRESHOLD = 20
EMA_PERIOD_1D = 50
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

def calculate_elder_ray(high, low, close, ema_period):
    """Calculate Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = calculate_ema(close, ema_period)
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def calculate_dmi(high, low, close, period):
    """Calculate DMI components for ADX"""
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
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    tr_sum = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).sum().values
    dm_plus_sum = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).sum().values
    dm_minus_sum = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend direction
    daily_close = df_daily['close'].values
    daily_ema = calculate_ema(daily_close, EMA_PERIOD_1D)
    
    # Align daily EMA to 6h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_daily, daily_ema)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Elder Ray
    bull_power, bear_power = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    
    # ADX for trend strength
    adx = calculate_dmi(high, low, close, ADX_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_PERIOD, ADX_PERIOD, EMA_PERIOD_1D) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(daily_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Trend filter: price relative to daily EMA
        above_daily_ema = close[i] > daily_ema_aligned[i]
        below_daily_ema = close[i] < daily_ema_aligned[i]
        
        # Entry conditions: strong Elder Ray power in direction of trend with ADX > threshold
        long_entry = (bull_power[i] > 0) and above_daily_ema and (adx[i] > ADX_THRESHOLD)
        short_entry = (bear_power[i] < 0) and below_daily_ema and (adx[i] > ADX_THRESHOLD)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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