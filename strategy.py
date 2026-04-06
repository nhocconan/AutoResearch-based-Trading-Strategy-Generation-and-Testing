#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour ADX + Elder Ray (Bull/Bear Power) with regime filter.
# Uses 12h EMA(50) for trend direction and Elder Ray for momentum.
# ADX > 25 filters for trending markets, Elder Ray crossovers for entry.
# Works in bull (strong bull power) and bear (strong bear power) markets.
# Target: 100-200 total trades over 4 years (25-50/year) to balance frequency and costs.
# Position size: 0.25, max drawdown controlled by ADX filter and stop loss.

name = "exp_13719_6h_adx_elderray12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
ELDER_RAY_PERIOD = 13
EMA_TREND_PERIOD = 50
VOLUME_MA_PERIOD = 10
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_adx(high, low, close, period):
    """Calculate ADX using Wilder's smoothing"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / tr_smooth
    minus_di = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_elder_ray(high, low, close, period):
    """Calculate Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = calculate_ema(close, period)
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for Elder Ray and EMA trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Elder Ray (Bull/Bear Power) and EMA trend
    bull_power, bear_power = calculate_elder_ray(high_12h, low_12h, close_12h, ELDER_RAY_PERIOD)
    ema_trend = calculate_ema(close_12h, EMA_TREND_PERIOD)
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    ema_trend_aligned = align_htf_to_ltf(prices, df_12h, ema_trend)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ADX for trend strength filter
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # Volume filter
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, ELDER_RAY_PERIOD, EMA_TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_trend_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 12h EMA
        uptrend = close[i] > ema_trend_aligned[i]
        downtrend = close[i] < ema_trend_aligned[i]
        
        # Elder Ray signals
        bull_power_cross = bull_power_aligned[i] > 0 and (i == 0 or bull_power_aligned[i-1] <= 0)
        bear_power_cross = bear_power_aligned[i] < 0 and (i == 0 or bear_power_aligned[i-1] >= 0)
        
        # Generate signals
        if position == 0:
            # Long: ADX > threshold, uptrend, bull power crosses above zero, volume confirmation
            if adx[i] > ADX_THRESHOLD and uptrend and bull_power_cross and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: ADX > threshold, downtrend, bear power crosses below zero, volume confirmation
            elif adx[i] > ADX_THRESHOLD and downtrend and bear_power_cross and volume_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bear power crosses below zero or trend change
            if bear_power_aligned[i] < 0 and (i == 0 or bear_power_aligned[i-1] >= 0):
                signals[i] = 0.0
                position = 0
            elif not uptrend:  # trend change to downtrend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short: bull power crosses above zero or trend change
            if bull_power_aligned[i] > 0 and (i == 0 or bull_power_aligned[i-1] <= 0):
                signals[i] = 0.0
                position = 0
            elif not downtrend:  # trend change to uptrend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals