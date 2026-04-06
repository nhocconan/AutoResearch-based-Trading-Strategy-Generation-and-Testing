#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Keltner Channel breakout with volume confirmation on 1d timeframe
# Keltner Channels (ATR-based) adapt to volatility, working in both trending and ranging markets.
# Breakouts above upper/lower bands with volume confirmation capture strong moves.
# Weekly timeframe provides structural context, reducing false signals.
# Target: 40-80 trades over 4 years (10-20/year) to minimize fee drag.

name = "exp_12904_1d_weekly_keltner_breakout_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
EMA_PERIOD = 20
ATR_PERIOD = 10
KC_MULTIPLIER = 1.5
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_keltner_channels(high, low, close, ema_period, atr_period, multiplier):
    """Calculate Keltner Channels"""
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    upper = ema + (multiplier * atr)
    lower = ema - (multiplier * atr)
    return ema, upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Keltner Channels
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    ema_w, upper_w, lower_w = calculate_keltner_channels(
        high_w, low_w, close_w, EMA_PERIOD, ATR_PERIOD, KC_MULTIPLIER
    )
    
    # Align to daily timeframe
    ema_aligned = align_htf_to_ltf(prices, df_weekly, ema_w)
    upper_aligned = align_htf_to_ltf(prices, df_weekly, upper_w)
    lower_aligned = align_htf_to_ltf(prices, df_weekly, lower_w)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, ATR_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Keltner levels not available
        if np.isnan(ema_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]):
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
        
        # Breakout above upper band or below lower band
        breakout_long = volume_ok and close[i] >= upper_aligned[i]
        breakout_short = volume_ok and close[i] <= lower_aligned[i]
        
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