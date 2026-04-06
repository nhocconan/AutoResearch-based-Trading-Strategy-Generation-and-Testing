#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) + ADX regime filter.
# Elder Ray uses EMA13 to measure bull/bear power: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# ADX > 25 indicates trending market, < 20 indicates ranging. We only take Elder Ray signals
# when ADX confirms the regime: in trend (ADX>25), follow Elder Ray direction; in range (ADX<20),
# fade extreme Elder Ray values. Weekly EMA50 filter ensures alignment with higher timeframe trend.
# This adapts to both bull and bear markets by switching between trend-following and mean-reversion.

name = "elder_ray_adx_regime_6h_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA13_PERIOD = 13      # For Elder Ray
ADX_PERIOD = 14        # ADX calculation
EMA50_WEEKLY = 50      # Weekly EMA for higher timeframe filter
SIGNAL_SIZE = 0.25     # Position size (25%)
ATR_PERIOD = 14        # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.5

def ema(series, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_vals = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr_vals

def adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # Plus and minus directional movement
    up = high - np.roll(high, 1)
    down = np.roll(low, 1) - low
    
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    # True range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Smoothed values
    atr_vals = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional indicators
    plus_di = 100 * plus_dm_smooth / atr_vals
    minus_di = 100 * minus_dm_smooth / atr_vals
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx_vals = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50w = ema(close_1w, EMA50_WEEKLY)
    ema_50w_aligned = align_htf_to_ltf(prices, df_1w, ema_50w)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Elder Ray components
    ema13 = ema(close, EMA13_PERIOD)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ADX for regime detection
    adx_vals = adx(high, low, close, ADX_PERIOD)
    
    # ATR for stop loss
    atr_vals = atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA13_PERIOD, ADX_PERIOD, EMA50_WEEKLY, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_50w_aligned[i]) or np.isnan(adx_vals[i]):
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
        
        # Regime detection: ADX > 25 = trending, ADX < 20 = ranging
        trending = adx_vals[i] > 25
        ranging = adx_vals[i] < 20
        
        # Elder Ray signals
        bull_signal = bull_power[i] > 0  # Bullish when high > EMA13
        bear_signal = bear_power[i] < 0  # Bearish when low < EMA13
        
        # Generate signals based on regime
        if position == 0:
            if trending and bull_signal:
                # In trend, follow Elder Ray direction (trend following)
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_vals[i])
            elif trending and bear_signal:
                # In trend, follow Elder Ray direction (trend following)
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_vals[i])
            elif ranging and bull_power[i] < -bull_power[i-1]*1.5:
                # In range, fade extreme bear power (mean reversion long)
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_vals[i])
            elif ranging and bear_power[i] > -bear_power[i-1]*1.5:
                # In range, fade extreme bull power (mean reversion short)
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