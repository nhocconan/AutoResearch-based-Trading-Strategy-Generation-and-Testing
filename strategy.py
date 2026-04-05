#!/usr/bin/env python3
"""
Experiment #9719: 6h Bollinger Band Mean Reversion with Volume Spike and Regime Filter.
Hypothesis: Bollinger Band mean reversion (touching bands) combined with volume spikes 
and regime filtering (ADX < 25 for mean reversion) provides high-probability reversals 
in ranging markets, while avoiding trend-following losses in strong trends. 
Works in bull/bear markets via mean reversion at extremes. Targets 75-150 total trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9719_6h_bb_meanrev_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD = 2.0
VOLUME_SPIKE_MULTIPLIER = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_true_range(high, low, close):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First value
    return tr

def calculate_adx(high, low, close, period):
    tr = calculate_true_range(high, low, close)
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Wilder's smoothing
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_atr(high, low, close, period):
    tr = calculate_true_range(high, low, close)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_bbands(close, period, std_dev):
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower, sma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = volumes = prices['volume'].values  # Fix: use correct column
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_middle = calculate_bbands(close, BB_PERIOD, BB_STD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX for regime filtering
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BB_PERIOD, 20, ADX_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Regime filter: only trade mean reversion in low volatility (ADX < 25)
        low_volatility = adx[i] < ADX_THRESHOLD
        
        # Mean reversion signals: touch Bollinger Bands with volume spike
        long_signal = low_volatility and volume_spike and close[i] <= bb_lower[i]
        short_signal = low_volatility and volume_spike and close[i] >= bb_upper[i]
        
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