#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour VWAP mean reversion with weekly trend filter and volume confirmation.
# Price tends to revert to VWAP in ranging markets (common in 2025-2026), while weekly EMA
# filters for trend alignment to avoid counter-trend trades. VWAP deviation >1.5σ triggers
# mean reversion entries. Target: 50-150 total trades over 4 years.

name = "exp_13348_12h_vwap_meanrev_weekly_trend_vol"
timeframe = "12h"
leverage = 1.0

# Parameters
VWAP_DEV_THRESHOLD = 1.5     # Standard deviations for entry
VWAP_EXIT_THRESHOLD = 0.5    # Standard deviations for exit
VWAP_LOOKBACK = 20           # Period for VWAP std dev calculation
WEEKLY_EMA_PERIOD = 20       # Weekly EMA for trend filter
VOLUME_MA_PERIOD = 20        # Volume moving average
VOLUME_THRESHOLD = 1.3       # Volume must be above average
SIGNAL_SIZE = 0.25           # Position size (25%)
ATR_PERIOD = 14              # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.0    # ATR multiplier for stop loss

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP for given period"""
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    return np.divide(vwap_numerator, vwap_denominator, 
                     out=np.full_like(vwap_numerator, np.nan), 
                     where=vwap_denominator!=0)

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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, WEEKLY_EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP and its standard deviation
    vwap = calculate_vwap(high, low, close, volume)
    vwap_dev = close - vwap
    
    # Calculate rolling standard deviation of VWAP deviation
    vwap_dev_series = pd.Series(vwap_dev)
    vwap_std = vwap_dev_series.rolling(window=VWAP_LOOKBACK, min_periods=VWAP_LOOKBACK).std().values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_EMA_PERIOD, VWAP_LOOKBACK, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(vwap_std[i]) or 
            np.isnan(vwap_dev[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter: price relative to weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # VWAP mean reversion signals
        vwap_zscore = vwap_dev[i] / vwap_std[i] if vwap_std[i] > 0 else 0
        
        mean_revert_long = volume_ok and uptrend and (vwap_zscore < -VWAP_DEV_THRESHOLD)
        mean_revert_short = volume_ok and downtrend and (vwap_zscore > VWAP_DEV_THRESHOLD)
        
        # Exit when price returns near VWAP
        exit_long = position == 1 and abs(vwap_zscore) < VWAP_EXIT_THRESHOLD
        exit_short = position == -1 and abs(vwap_zscore) < VWAP_EXIT_THRESHOLD
        
        # Generate signals
        if position == 0:
            if mean_revert_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif mean_revert_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals