#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian breakout with daily volume confirmation and weekly trend filter
# Works in bull/bear markets because breakouts capture momentum, volume filters false signals,
# and weekly trend alignment ensures trades are in direction of higher timeframe momentum.
# Target: 80-150 total trades over 4 years (20-38/year) to balance opportunity and cost.

name = "exp_12987_6h_donchian20_1d_vol_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
TREND_EMA_PERIOD = 50

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily and weekly data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate daily volume MA
    volume_daily = df_daily['volume'].values
    volume_ma_daily = pd.Series(volume_daily).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Calculate weekly EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema_weekly = calculate_ema(close_weekly, TREND_EMA_PERIOD)
    
    # Align to 6h timeframe
    volume_ma_aligned = align_htf_to_ltf(prices, df_daily, volume_ma_daily)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    upper, lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(volume_ma_aligned[i]) or np.isnan(ema_weekly_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
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
        volume_ok = volume[i] > (volume_ma_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_aligned[i]) else False
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_weekly_aligned[i]
        downtrend = close[i] < ema_weekly_aligned[i]
        
        # Breakout conditions
        breakout_up = volume_ok and uptrend and close[i] >= upper[i]
        breakout_down = volume_ok and downtrend and close[i] <= lower[i]
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
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