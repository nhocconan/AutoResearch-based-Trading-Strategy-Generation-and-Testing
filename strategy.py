#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily donchian breakout with 1d volume confirmation and 1w trend filter
# Works in bull/bear because breakouts capture strong moves, volume filters weak signals,
# and weekly EMA filter ensures we only trade in the direction of the higher timeframe trend.
# Target: 80-150 trades over 4 years (20-38/year) to balance opportunity and fee cost.

name = "exp_12877_4h_donchian20_1d_vol_1wtrend_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
EMA_TREND_PERIOD = 50
SIGNAL_SIZE = 0.25
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

def calculate_ema(values, period):
    """Calculate EMA"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate daily donchian channels
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    volume_d = df_daily['volume'].values
    
    donchian_high = pd.Series(high_d).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low_d).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Calculate daily volume MA
    volume_ma = pd.Series(volume_d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Calculate weekly EMA for trend filter
    close_w = df_weekly['close'].values
    ema_w = calculate_ema(close_w, EMA_TREND_PERIOD)
    
    # Calculate ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Align daily indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    volume_ma_aligned = align_htf_to_ltf(prices, df_daily, volume_ma)
    ema_w_aligned = align_htf_to_ltf(prices, df_weekly, ema_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, EMA_TREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(volume_ma_aligned[i]) or np.isnan(ema_w_aligned[i]):
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
        volume_ok = volume_d[i] > (volume_ma_aligned[i] * VOLUME_THRESHOLD) if i < len(volume_d) and not np.isnan(volume_ma_aligned[i]) else False
        
        # Determine trend direction from weekly EMA
        uptrend = close_w[i] > ema_w[i] if i < len(close_w) and not np.isnan(ema_w[i]) else False
        downtrend = close_w[i] < ema_w[i] if i < len(close_w) and not np.isnan(ema_w[i]) else False
        
        # Breakout signals with trend filter
        breakout_long = volume_ok and close[i] >= donchian_high_aligned[i] and uptrend
        breakout_short = volume_ok and close[i] <= donchian_low_aligned[i] and downtrend
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
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