#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot levels (from daily) with volume confirmation and Choppiness regime filter.
# Camarilla levels provide high-probability reversal zones; volume confirms institutional interest.
# Choppiness filter avoids whipsaws in strong trends. Works in bull/bear by fading extremes.
# Target: 80-160 total trades over 4 years (20-40/year) with ~0.25 position size.

name = "exp_13223_4h_camarilla1d_vol_chop_v2"
timeframe = "4h"
leverage = 1.0

# Parameters
CAMARILLA_LOOKBACK = 1  # Use prior day's OHLC
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
CHOPPINESS_PERIOD = 14
CHOPPINESS_TREND_THRESHOLD = 38.2  # Below = trending
CHOPPINESS_RANGE_THRESHOLD = 61.8  # Above = ranging
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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
    atr_sum = pd.Series(calculate_atr(high, low, close, 1)).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.fillna(50).values  # Neutral when undefined

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on prior day's range)
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + (range_1d * 1.1 / 2)  # H4 resistance
    camarilla_l4 = close_1d - (range_1d * 1.1 / 2)  # L4 support
    camarilla_h3 = close_1d + (range_1d * 1.1 / 4)  # H3 resistance
    camarilla_l3 = close_1d - (range_1d * 1.1 / 4)  # L3 support
    
    # Align Camarilla levels to 4h timeframe (shifted by 1 day for prior day's data)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, CHOPPINESS_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, CHOPPINESS_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available (first day)
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]):
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
        
        # Choppiness filter: only trade in ranging markets (avoid strong trends)
        chop_ok = chop[i] > CHOPPINESS_RANGE_THRESHOLD
        
        # Mean reversion signals at Camarilla levels
        # Long near L3/L4 with rejection, Short near H3/H4 with rejection
        long_signal = volume_ok and chop_ok and (
            (low[i] <= l3_aligned[i] and close[i] > l3_aligned[i]) or  # Bounce off L3
            (low[i] <= l4_aligned[i] and close[i] > l4_aligned[i])   # Bounce off L4
        )
        short_signal = volume_ok and chop_ok and (
            (high[i] >= h3_aligned[i] and close[i] < h3_aligned[i]) or  # Rejection at H3
            (high[i] >= h4_aligned[i] and close[i] < h4_aligned[i])   # Rejection at H4
        )
        
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