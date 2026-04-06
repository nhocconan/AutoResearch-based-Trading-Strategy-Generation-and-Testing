#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Daily Trend Filter + Volume Spike
# Elder Ray (bull power = high - EMA13, bear power = EMA13 - low) captures institutional buying/selling pressure.
# Daily trend filter (price > EMA50) ensures trades align with higher timeframe direction.
# Volume spike (>2x 20-period average) confirms institutional participation.
# Works in bull/bear because it follows institutional flow with trend alignment.
# Target: 80-150 trades over 4 years (20-38/year) for statistical validity.

name = "exp_12927_6h_elderay_daily_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA13_PERIOD = 13
EMA50_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
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

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend filter
    close_d = df_daily['close'].values
    ema50_d = calculate_ema(close_d, EMA50_PERIOD)
    ema50_d_aligned = align_htf_to_ltf(prices, df_daily, ema50_d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    ema13 = calculate_ema(close, EMA13_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA13_PERIOD, EMA50_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily EMA50 not available
        if np.isnan(ema50_d_aligned[i]):
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
        
        # Elder Ray components
        bull_power = high[i] - ema13[i]  # High - EMA13
        bear_power = ema13[i] - low[i]   # EMA13 - Low
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter from daily timeframe
        uptrend = close[i] > ema50_d_aligned[i]
        downtrend = close[i] < ema50_d_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: bull power > bear power (buying pressure) + uptrend + volume spike
            if bull_power > bear_power and uptrend and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: bear power > bull power (selling pressure) + downtrend + volume spike
            elif bear_power > bull_power and downtrend and volume_ok:
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