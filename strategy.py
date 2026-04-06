#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12711_6d_trix_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
Trix_PERIOD = 9
Trix_SIGNAL_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
CHOP_PERIOD = 14
CHOP_THRESHOLD = 61.8  # >61.8 = ranging, <38.2 = trending

def calculate_trix(close, period):
    """Calculate TRIX: triple EMA of % change"""
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    pct_change = ema3.pct_change()
    trix = pct_change.ewm(span=Trix_SIGNAL_PERIOD, adjust=False, min_periods=Trix_SIGNAL_PERIOD).mean() * 100
    return trix.values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_chop(high, low, close, period):
    """Calculate Choppiness Index"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily TRIX
    close_1d = df_1d['close'].values
    trix_1d = calculate_trix(close_1d, Trix_PERIOD)
    
    # Align TRIX to 6h timeframe
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    chop = calculate_chop(high, low, close, CHOP_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, Trix_PERIOD + Trix_SIGNAL_PERIOD, CHOP_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if TRIX not available
        if np.isnan(trix_1d_aligned[i]) or np.isnan(chop[i]):
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
        
        # TRIX momentum + Chop regime filter
        # In ranging markets (CHOP > 61.8): fade TRIX extremes
        # In trending markets (CHOP < 38.2): follow TRIX direction
        trix = trix_1d_aligned[i]
        ranging = chop[i] > CHOP_THRESHOLD
        trending = chop[i] < (100 - CHOP_THRESHOLD)  # 38.2
        
        # Fade signals in ranging market: sell high TRIX, buy low TRIX
        fade_long = volume_ok and ranging and trix < -0.5
        fade_short = volume_ok and ranging and trix > 0.5
        
        # Trend signals in trending market: buy rising TRIX, sell falling TRIX
        # Need previous TRIX to check direction
        if i > start:
            prev_trix = trix_1d_aligned[i-1]
            trix_rising = trix > prev_trix
            trix_falling = trix < prev_trix
        else:
            trix_rising = False
            trix_falling = False
            
        trend_long = volume_ok and trending and trix_rising and trix > 0
        trend_short = volume_ok and trending and trix_falling and trix < 0
        
        # Entry conditions
        long_entry = fade_long or trend_long
        short_entry = fade_short or trend_short
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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