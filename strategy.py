#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with daily ADX regime filter
# Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# ADX > 25 = trending (follow Elder Ray), ADX < 20 = ranging (fade Elder Ray extremes)
# Works in bull/bear because it adapts to regime: trend following in trends, mean reversion in ranges.
# Target: 80-180 trades over 4 years (20-45/year) to balance signal quality and frequency.

name = "exp_12991_6h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA_PERIOD = 13
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
BULL_BEAR_THRESHOLD = 0.0  # Zero line for Elder Ray
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA with proper Wilder's smoothing"""
    return pd.Series(close).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

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
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily indicators
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # Daily EMA for Elder Ray
    ema_d = calculate_ema(close_d, EMA_PERIOD)
    
    # Daily Bull Power and Bear Power
    bull_power = high_d - ema_d
    bear_power = ema_d - low_d
    
    # Daily ADX
    adx_d = calculate_adx(high_d, low_d, close_d, ADX_PERIOD)
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_daily, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_daily, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx_d)
    
    # Calculate 6h ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, ADX_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(adx_aligned[i]):
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
        
        # Determine regime and signal
        adx = adx_aligned[i]
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        
        if adx > ADX_TREND_THRESHOLD:  # Trending regime
            # Follow Elder Ray: long when bull power positive, short when bear power positive
            if bull > BULL_BEAR_THRESHOLD and bear <= BULL_BEAR_THRESHOLD:
                signal_val = SIGNAL_SIZE
            elif bear > BULL_BEAR_THRESHOLD and bull <= BULL_BEAR_THRESHOLD:
                signal_val = -SIGNAL_SIZE
            else:
                signal_val = 0.0
        elif adx < ADX_RANGE_THRESHOLD:  # Ranging regime
            # Fade Elder Ray extremes: short when bull power too high, long when bear power too high
            if bull > 0 and bear <= 0:  # Overbought - fade
                signal_val = -SIGNAL_SIZE
            elif bear > 0 and bull <= 0:  # Oversold - fade
                signal_val = SIGNAL_SIZE
            else:
                signal_val = 0.0
        else:  # Transition zone (20 <= ADX <= 25)
            signal_val = 0.0
        
        # Handle position changes
        if signal_val == 0.0:
            signals[i] = 0.0
            position = 0
        elif signal_val > 0:  # Long signal
            if position != 1:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = SIGNAL_SIZE
        else:  # Short signal
            if position != -1:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals