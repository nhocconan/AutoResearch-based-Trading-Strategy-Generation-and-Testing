#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot levels from 1-day timeframe with volume spike confirmation and choppiness regime filter.
# In bull markets, buy near L3 support and sell at H3 resistance; in bear markets, sell near H3 resistance and buy at L3 support.
# The daily Camarilla levels provide institutional support/resistance levels, volume spike confirms institutional interest,
# and choppiness filter avoids ranging markets where pivot levels fail. Target: 75-200 total trades over 4 years (19-50/year).

name = "exp_13200_4h_camarilla1d_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day's OHLC for Camarilla calculation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
CHOPPINESS_PERIOD = 14
CHOPPINESS_THRESHOLD = 38.2  # Below this = trending (use pivot breakouts)
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

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    atr_sum = pd.Series(high - low).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.fillna(50).values  # Neutral when undefined

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close
    
    # Camarilla levels
    H4 = close + range_val * 1.1 / 2
    H3 = close + range_val * 1.1 / 4
    L3 = close - range_val * 1.1 / 4
    L4 = close - range_val * 1.1 / 2
    return H4, H3, L3, L4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use only completed daily bars
    H4_1d = np.roll(high_1d, 1)
    L4_1d = np.roll(low_1d, 1)
    C_1d = np.roll(close_1d, 1)
    
    # Calculate Camarilla levels for each day
    H3_1d = np.zeros_like(high_1d)
    L3_1d = np.zeros_like(high_1d)
    for i in range(len(high_1d)):
        if i == 0:
            H3_1d[i] = np.nan
            L3_1d[i] = np.nan
        else:
            _, H3_1d[i], L3_1d[i], _ = calculate_camarilla(H4_1d[i], L4_1d[i], C_1d[i])
    
    # Align Camarilla levels to 4h timeframe
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    
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
    start = max(PIVOT_LOOKBACK + 1, VOLUME_MA_PERIOD, ATR_PERIOD, CHOPPINESS_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]):
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
        
        # Choppiness filter: only trade in trending markets (chop < threshold)
        chop_ok = chop[i] < CHOPPINESS_THRESHOLD if not np.isnan(chop[i]) else False
        
        # Breakout signals using Camarilla levels
        breakout_up = volume_ok and chop_ok and (high[i] > H3_1d_aligned[i])
        breakout_down = volume_ok and chop_ok and (low[i] < L3_1d_aligned[i])
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
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