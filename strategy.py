#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (Bull Power/Bear Power) with weekly EMA trend filter and volume confirmation.
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13). 
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with volume confirmation and weekly uptrend.
# Short when Bear Power < 0 and falling, Bull Power < 0 and rising, with volume confirmation and weekly downtrend.
# This captures institutional buying/selling pressure in both bull and bear markets.
# Target: 75-175 total trades over 4 years (19-44/year).

name = "exp_13315_6h_elder_ray_weekly_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_PERIOD = 13  # EMA for Elder Ray
WEEKLY_EMA_PERIOD = 21  # Weekly trend filter
VOLUME_MA_PERIOD = 20   # Volume confirmation
VOLUME_THRESHOLD = 1.5  # Volume must be 1.5x average
SIGNAL_SIZE = 0.25      # Position size (25%)
ATR_PERIOD = 14         # ATR for stoploss
ATR_STOP_MULTIPLIER = 2.5  # Stoploss at 2.5x ATR

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
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
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, WEEKLY_EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components: EMA(13) of close
    ema_13 = calculate_ema(close, ELDER_RAY_PERIOD)
    bull_power = high - ema_13  # High - EMA
    bear_power = low - ema_13   # Low - EMA
    
    # Volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_PERIOD, WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_1w_aligned[i]):
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
        
        # Volume confirmation (need at least 2 periods to check rising/falling)
        if i < 2:
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: weekly EMA direction
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Elder Ray signals: check if bull/bear power is rising/falling
        bull_rising = bull_power[i] > bull_power[i-1]
        bull_falling = bull_power[i] < bull_power[i-1]
        bear_rising = bear_power[i] > bear_power[i-1]
        bear_falling = bear_power[i] < bear_power[i-1]
        
        # Long conditions: Bull power positive AND rising, with volume and uptrend
        long_signal = (bull_power[i] > 0) and bull_rising and volume_ok and uptrend
        # Short conditions: Bear power negative AND falling, with volume and downtrend
        short_signal = (bear_power[i] < 0) and bear_falling and volume_ok and downtrend
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
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