#!/usr/bin/env python3
"""
Experiment #10151: 6h Time-Weighted Average Price (TWAP) Deviation + Daily Trend + Volume
Hypothesis: Price deviations from daily TWAP (volume-weighted average price) tend to revert toward the mean when the daily trend (EMA50) is strong, but continue in the direction of the trend when deviation is extreme with volume confirmation. Works in both bull and bear markets by fading moderate deviations and catching strong trending moves. Target: 80-160 total trades over 4 years (20-40/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10151_6h_twap_deviation_daily_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TWAP_LOOKBACK = 24  # 24 * 6h = 6 days for daily VWAP approximation
TWAP_DEV_THRESHOLD_LOW = 0.015   # 1.5% deviation for mean reversion
TWAP_DEV_THRESHOLD_HIGH = 0.04   # 4% deviation for trend continuation
DAILY_EMA_PERIOD = 50
VOLUME_SPIKE_MULTIPLIER = 1.8
SIGNAL_SIZE_MEAN = 0.25   # Mean reversion position
SIGNAL_SIZE_TREND = 0.35  # Trend continuation position
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_vwap(high, low, close, volume, window):
    """Calculate Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3.0
    vwap_num = np.convolve(typical_price * volume, np.ones(window), 'same')
    vwap_den = np.convolve(volume, np.ones(window), 'same')
    # Avoid division by zero
    vwap_den[vwap_den == 0] = 1e-10
    vwap = vwap_num / vwap_den
    # Handle edges
    vwap[:window-1] = vwap[window-1]
    vwap[-window+1:] = vwap[-window+1]
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
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for VWAP and trend
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily VWAP (approximation)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    daily_volume = df_daily['volume'].values
    
    # Daily VWAP (using typical price * volume / volume)
    daily_typical = (daily_high + daily_low + daily_close) / 3.0
    vwap_num = np.convolve(daily_typical * daily_volume, np.ones(TWAP_LOOKBACK), 'same')
    vwap_den = np.convolve(daily_volume, np.ones(TWAP_LOOKBACK), 'same')
    vwap_den[vwap_den == 0] = 1e-10
    daily_vwap = vwap_num / vwap_den
    # Handle edges for convolution
    daily_vwap[:TWAP_LOOKBACK-1] = daily_vwap[TWAP_LOOKBACK-1]
    daily_vwap[-TWAP_LOOKBACK+1:] = daily_vwap[-TWAP_LOOKBACK+1]
    
    # Calculate daily EMA for trend direction
    daily_ema = calculate_ema(daily_close, DAILY_EMA_PERIOD)
    
    # Align daily indicators to 6h timeframe
    daily_vwap_aligned = align_htf_to_ltf(prices, df_daily, daily_vwap)
    daily_ema_aligned = align_htf_to_ltf(prices, df_daily, daily_ema)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TWAP_LOOKBACK, DAILY_EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily indicators not available
        if np.isnan(daily_vwap_aligned[i]) or np.isnan(daily_ema_aligned[i]):
            signals[i] = position * (SIGNAL_SIZE_MEAN if position > 0 else SIGNAL_SIZE_TREND) if position != 0 else 0.0
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
        
        # Calculate TWAP deviation percentage
        twap_dev = (close[i] - daily_vwap_aligned[i]) / daily_vwap_aligned[i]
        
        # Volume spike confirmation
        volume_ma = np.mean(volume[max(0, i-19):i+1]) if i >= 20 else volume[i]
        volume_spike = volume[i] > (volume_ma * VOLUME_SPIKE_MULTIPLIER) if volume_ma > 0 else False
        
        # Trend filter: price above/below daily EMA
        above_daily_ema = close[i] > daily_ema_aligned[i]
        below_daily_ema = close[i] < daily_ema_aligned[i]
        
        # Mean reversion: fade moderate deviations
        mean_reversion_long = (twap_dev < -TWAP_DEV_THRESHOLD_LOW) and above_daily_ema and not volume_spike
        mean_reversion_short = (twap_dev > TWAP_DEV_THRESHOLD_LOW) and below_daily_ema and not volume_spike
        
        # Trend continuation: extreme deviations with volume
        trend_continuation_long = (twap_dev > TWAP_DEV_THRESHOLD_HIGH) and above_daily_ema and volume_spike
        trend_continuation_short = (twap_dev < -TWAP_DEV_THRESHOLD_HIGH) and below_daily_ema and volume_spike
        
        # Generate signals
        if position == 0:
            if mean_reversion_long:
                signals[i] = SIGNAL_SIZE_MEAN
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif mean_reversion_short:
                signals[i] = -SIGNAL_SIZE_MEAN
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif trend_continuation_long:
                signals[i] = SIGNAL_SIZE_TREND
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif trend_continuation_short:
                signals[i] = -SIGNAL_SIZE_TREND
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE_MEAN if position > 0 else SIGNAL_SIZE_TREND
        elif position == -1:
            signals[i] = -SIGNAL_SIZE_MEAN if position > 0 else -SIGNAL_SIZE_TREND
    
    return signals