#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly volatility breakout with volume confirmation and ATR stoploss
# Works in bull/bear because volatility expansion precedes strong moves in either direction.
# Volume filters out false breakouts. Weekly timeframe provides structural context.
# Target: 60-100 trades over 4 years (15-25/year) to balance opportunity and cost.

name = "exp_12978_1d_weekly_volatility_breakout_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
VOLATILITY_LOOKBACK = 20
VOLATILITY_THRESHOLD = 1.5  # ATR ratio threshold for volatility expansion

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_volatility_ratio(high, low, close, period):
    """Calculate current ATR vs long-term ATR ratio"""
    atr_short = calculate_atr(high, low, close, period)
    atr_long = pd.Series(atr_short).ewm(span=period*2, adjust=False, min_periods=period*2).mean().values
    # Avoid division by zero
    ratio = np.where(atr_long > 0, atr_short / atr_long, 1.0)
    return ratio

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly ATR for volatility measurement
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    atr_weekly = calculate_atr(high_w, low_w, close_w, ATR_PERIOD)
    vol_ratio_weekly = calculate_volatility_ratio(high_w, low_w, close_w, ATR_PERIOD)
    
    # Align to daily timeframe
    atr_aligned = align_htf_to_ltf(prices, df_weekly, atr_weekly)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_weekly, vol_ratio_weekly)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr_daily = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, VOLATILITY_LOOKBACK) + 1
    
    for i in range(start, n):
        # Skip if volatility data not available
        if np.isnan(vol_ratio_aligned[i]) or np.isnan(atr_aligned[i]):
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
        
        # Volatility expansion filter
        vol_expansion = vol_ratio_aligned[i] > VOLATILITY_THRESHOLD
        
        # Breakout signals: price movement beyond weekly ATR
        price_change = abs(close[i] - close[i-1]) if i > 0 else 0
        breakout_threshold = atr_aligned[i]
        
        breakout_up = volume_ok and vol_expansion and (close[i] > close[i-1]) and (price_change > breakout_threshold)
        breakout_down = volume_ok and vol_expansion and (close[i] < close[i-1]) and (price_change > breakout_threshold)
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_daily[i])
            elif breakout_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_daily[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals