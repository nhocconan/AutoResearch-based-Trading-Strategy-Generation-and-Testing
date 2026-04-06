#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels with 12h trend filter and volume confirmation
# Camarilla levels from daily data provide high-probability reversal/continuation points
# Fade at R3/S3 (reversion to mean) and breakout continuation at R4/S4
# 12h EMA filter ensures alignment with higher timeframe trend
# Volume confirmation filters false breakouts
# Designed for 60-120 trades over 4 years (15-30/year) to balance opportunity and cost

name = "exp_13079_6h_camarilla12h_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use daily OHLC
EMA_FAST = 20
EMA_SLOW = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
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

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_ = high - low
    # Camarilla levels
    R4 = close + range_ * 1.1 / 2
    R3 = close + range_ * 1.1 / 4
    S3 = close - range_ * 1.1 / 4
    S4 = close - range_ * 1.1 / 2
    return R4, R3, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla calculation ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_R4 = np.zeros(len(close_1d))
    camarilla_R3 = np.zeros(len(close_1d))
    camarilla_S3 = np.zeros(len(close_1d))
    camarilla_S4 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        R4, R3, S3, S4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        camarilla_R4[i] = R4
        camarilla_R3[i] = R3
        camarilla_S3[i] = S3
        camarilla_S4[i] = S4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    
    # Load 12h data for trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_fast_12h = calculate_ema(close_12h, EMA_FAST)
    ema_slow_12h = calculate_ema(close_12h, EMA_SLOW)
    ema_fast_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_fast_12h)
    ema_slow_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_slow_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_FAST, EMA_SLOW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_fast_12h_aligned[i]) or np.isnan(ema_slow_12h_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if Camarilla levels not available
        if np.isnan(camarilla_R4_aligned[i]) or np.isnan(camarilla_S4_aligned[i]):
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
        
        # Trend filter: 12h EMA crossover
        uptrend = ema_fast_12h_aligned[i] > ema_slow_12h_aligned[i]
        downtrend = ema_fast_12h_aligned[i] < ema_slow_12h_aligned[i]
        
        # Camarilla signals
        # Fade at R3/S3 (mean reversion)
        fade_long = volume_ok and not uptrend and close[i] <= camarilla_S3_aligned[i] and close[i] > camarilla_S4_aligned[i]
        fade_short = volume_ok and not downtrend and close[i] >= camarilla_R3_aligned[i] and close[i] < camarilla_R4_aligned[i]
        
        # Breakout continuation at R4/S4
        breakout_long = volume_ok and uptrend and close[i] >= camarilla_R4_aligned[i]
        breakout_short = volume_ok and downtrend and close[i] <= camarilla_S4_aligned[i]
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long:
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