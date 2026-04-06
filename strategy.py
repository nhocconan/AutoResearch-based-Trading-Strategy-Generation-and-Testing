#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot reversal with daily volume confirmation and Choppiness regime filter.
# Uses Camarilla levels (H3/L3) from daily pivots for mean-reversion entries in ranging markets.
# Volume confirmation filters weak signals, Choppiness index avoids trending regimes.
# Works in both bull and bear markets by fading extremes during consolidation.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13516_12h_camarilla_1d_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
CHOPPINESS_PERIOD = 14
CHOPPINESS_THRESHOLD = 61.8  # >61.8 = ranging (good for mean reversion)
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    H3 = close + (range_val * 1.1 / 6)
    L3 = close - (range_val * 1.1 / 6)
    return H3, L3

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    atr1 = np.abs(high - low)
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(atr1, atr2), atr3)
    
    sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    chop = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(period)
    return chop.fillna(50).values  # fillna with 50 (neutral)

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    H3, L3 = calculate_camarilla(high_1d, low_1d, close_1d)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate daily Choppiness Index
    chop = calculate_choppiness(high_1d, low_1d, close_1d, CHOPPINESS_PERIOD)
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla or Choppiness not available
        if np.isnan(H3_12h[i]) or np.isnan(L3_12h[i]) or np.isnan(chop_12h[i]):
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
        
        # Choppiness filter: only trade in ranging markets (>61.8 = ranging)
        ranging_market = chop_12h[i] > CHOPPINESS_THRESHOLD
        
        # Mean reversion signals at Camarilla H3/L3 levels
        sell_signal = volume_ok and ranging_market and (high[i] >= H3_12h[i])
        buy_signal = volume_ok and ranging_market and (low[i] <= L3_12h[i])
        
        # Generate signals
        if position == 0:
            if sell_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif buy_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals