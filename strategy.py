#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Trix + Volume Spike + Choppiness Regime Filter
# Uses TRIX (12-period) for momentum, volume confirmation for institutional interest,
# and Choppiness Index to identify trending vs ranging markets.
# In trending markets (CHOP < 38.2): TRIX crossover signals
# In ranging markets (CHOP > 61.8): Mean reversion at Bollinger Bands
# This dual approach adapts to market conditions, working in both bull and bear markets.
# Target: 50-150 total trades over 4 years with strict entry conditions.

name = "exp_13362_12h_trix_volume_chop_regime_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
TRIX_PERIOD = 12
TRIX_SIGNAL = 9
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
CHOPPINESS_PERIOD = 14
CHOPPINESS_THRESHOLD_TREND = 38.2
CHOPPINESS_THRESHOLD_RANGE = 61.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
BBANDS_PERIOD = 20
BBANDS_STD = 2.0

def calculate_trix(close, period, signal):
    """Calculate TRIX and signal line"""
    # Triple exponential moving average
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    # TRIX = percent change of ema3
    trix = (ema3 / ema3.shift(1) - 1) * 100
    # Signal line
    trix_signal = trix.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return trix.values, trix_signal.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    atr = calculate_atr(high, low, close, 1)  # True Range
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(period)
    return chop.values

def calculate_bollinger_bands(close, period, std_dev):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean()
    std = pd.Series(close).rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper.values, lower.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX and signal line
    trix, trix_signal = calculate_trix(close, TRIX_PERIOD, TRIX_SIGNAL)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, CHOPPINESS_PERIOD)
    
    # Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(close, BBANDS_PERIOD, BBANDS_STD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TRIX_PERIOD + TRIX_SIGNAL, VOLUME_MA_PERIOD, ATR_PERIOD, CHOPPINESS_PERIOD, BBANDS_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if any indicator not available
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or 
            np.isnan(chop[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
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
        
        # Market regime detection
        is_trending = chop[i] < CHOPPINESS_THRESHOLD_TREND
        is_ranging = chop[i] > CHOPPINESS_THRESHOLD_RANGE
        
        # Generate signals based on regime
        if position == 0:
            if is_trending and volume_ok:
                # In trending markets: TRIX crossover
                if trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]:
                    # Bullish crossover
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]:
                    # Bearish crossover
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif is_ranging and volume_ok:
                # In ranging markets: mean reversion at Bollinger Bands
                if close[i] <= bb_lower[i]:
                    # Oversold - go long
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif close[i] >= bb_upper[i]:
                    # Overbought - go short
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