#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot levels from daily data with volume confirmation and volatility filter.
# Uses prior day's high/low/close to calculate Camarilla levels (resistance/support).
# Long when price touches S3/S4 with volume spike, short when touches R3/R4 with volume spike.
# Volatility filter (ATR ratio) avoids choppy markets. Designed for 12h timeframe to reduce trade frequency.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13162_12h_camarilla_pivot_1d_vol_volat_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard multiplier for R3/S3 etc.
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
VOLATILITY_LOOKBACK = 20
VOLATILITY_THRESHOLD = 0.5  # ATR ratio threshold for volatility filter
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR for volatility filter (using daily high/low/close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    atr_1d_ma = pd.Series(atr_1d).rolling(window=VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK).mean().values
    atr_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_ma)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: 
    # H4 = Close + 1.5*(High-Low)
    # H3 = Close + 1.1*(High-Low)
    # H2 = Close + 0.55*(High-Low)
    # H1 = Close + 0.275*(High-Low)
    # L1 = Close - 0.275*(High-Low)
    # L2 = Close - 0.55*(High-Low)
    # L3 = Close - 1.1*(High-Low)
    # L4 = Close - 1.5*(High-Low)
    # We'll use H3/L3 and H4/L4 for entries
    high_shifted = np.roll(high_1d, 1)  # Previous day's high
    low_shifted = np.roll(low_1d, 1)    # Previous day's low
    close_shifted = np.roll(close_1d, 1) # Previous day's close
    
    # Calculate Camarilla levels
    hl_range = high_shifted - low_shifted
    h3 = close_shifted + CAMARILLA_MULT * hl_range
    l3 = close_shifted - CAMARILLA_MULT * hl_range
    h4 = close_shifted + 1.5 * hl_range
    l4 = close_shifted - 1.5 * hl_range
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stoploss and volatility filter
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, VOLATILITY_LOOKBACK) + 1
    
    for i in range(start, n):
        # Skip if data not available
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(atr_1d_ma_aligned[i]) or np.isnan(volume_ma[i]):
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
        
        # Volatility filter: avoid low volatility (choppy) markets
        # Only trade when current ATR is above average ATR (enough volatility)
        vol_filter = atr[i] > atr_1d_ma_aligned[i] * VOLATILITY_THRESHOLD if not np.isnan(atr_1d_ma_aligned[i]) else False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Proximity to Camarilla levels (within 0.1% of level)
        proximity_threshold = 0.001  # 0.1%
        near_h3 = abs(high[i] - h3_aligned[i]) / h3_aligned[i] <= proximity_threshold
        near_h4 = abs(high[i] - h4_aligned[i]) / h4_aligned[i] <= proximity_threshold
        near_l3 = abs(low[i] - l3_aligned[i]) / l3_aligned[i] <= proximity_threshold
        near_l4 = abs(low[i] - l4_aligned[i]) / l4_aligned[i] <= proximity_threshold
        
        # Generate signals
        if position == 0:
            # Long when price touches L3/L4 with volume and volatility
            if vol_filter and volume_ok and (near_l3 or near_l4):
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short when price touches H3/H4 with volume and volatility
            elif vol_filter and volume_ok and (near_h3 or near_h4):
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