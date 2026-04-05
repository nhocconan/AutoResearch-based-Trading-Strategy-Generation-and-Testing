#!/usr/bin/env python3
"""
Experiment #9835: 6h Keltner Channel + Weekly RSI + Volume Spike
Hypothesis: In mean-reverting markets, price touching Keltner Channel bands with weekly RSI extremes 
and volume spikes provides high-probability reversals. Works in both bull/bear via RSI regime filter.
Target: 80-160 total trades over 4 years (20-40/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9835_6h_keltner_rsi_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
KC_EMA_PERIOD = 20
KC_ATR_MULT = 2.0
KC_ATR_PERIOD = 10
WEEKLY_RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_SPIKE_MULT = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULT = 2.0

def calculate_ema(close, period):
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_rsi(close, period):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
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
    
    # Load weekly data ONCE
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_rsi = calculate_rsi(weekly_close, WEEKLY_RSI_PERIOD)
    weekly_rsi_aligned = align_htf_to_ltf(prices, df_weekly, weekly_rsi)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel
    ema_mid = calculate_ema(close, KC_EMA_PERIOD)
    atr_kc = calculate_atr(high, low, close, KC_ATR_PERIOD)
    kc_upper = ema_mid + (KC_ATR_MULT * atr_kc)
    kc_lower = ema_mid - (KC_ATR_MULT * atr_kc)
    
    # Volume MA for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stops
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    start = max(KC_EMA_PERIOD, KC_ATR_PERIOD, WEEKLY_RSI_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if weekly RSI not available
        if np.isnan(weekly_rsi_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume spike
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULT) if not np.isnan(volume_ma[i]) else False
        
        # Weekly RSI regime
        rsi_overbought = weekly_rsi_aligned[i] > RSI_OVERBOUGHT
        rsi_oversold = weekly_rsi_aligned[i] < RSI_OVERSOLD
        
        # Keltner touch conditions
        touch_upper = close[i] >= kc_upper[i] if not np.isnan(kc_upper[i]) else False
        touch_lower = close[i] <= kc_lower[i] if not np.isnan(kc_lower[i]) else False
        
        # Mean reversion entries: fade extremes with volume
        long_entry = touch_lower and rsi_oversold and volume_spike
        short_entry = touch_upper and rsi_overbought and volume_spike
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULT * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULT * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals