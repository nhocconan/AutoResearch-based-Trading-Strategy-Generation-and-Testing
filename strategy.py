#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal strategy with volume confirmation and volatility filter.
# Goes long at Camarilla L3 support with volume spike in non-volatile market,
# short at H3 resistance with volume spike in non-volatile market.
# Uses 1-day Camarilla levels (calculated from prior day OHLC) as key support/resistance.
# Works in both bull and bear markets by fading extremes at statistically significant levels.
# Volume confirmation ensures institutional participation, volatility filter avoids choppy markets.
# Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

name = "exp_13786_4h_camarilla_vol_volfilt_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
CAMARILLA_LOOKBACK = 1  # Use previous day for pivot calculation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
VOLATILITY_LOOKBACK = 20
VOLATILITY_THRESHOLD = 0.02  # 2% daily ATR as percentage of price
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    # Typical price
    typical = (high + low + close) / 3
    # Range
    range_val = high - low
    
    # Camarilla levels
    H4 = close + range_val * 1.1 / 2
    H3 = close + range_val * 1.1 / 4
    H2 = close + range_val * 1.1 / 6
    H1 = close + range_val * 1.1 / 12
    L1 = close - range_val * 1.1 / 12
    L2 = close - range_val * 1.1 / 6
    L3 = close - range_val * 1.1 / 4
    L4 = close - range_val * 1.1 / 2
    
    return H3, L3, H4, L4  # Return key levels for trading

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and volatility filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    camarilla_H3, camarilla_L3, camarilla_H4, camarilla_L4 = calculate_camarilla(high_1d, low_1d, close_1d)
    camarilla_H3 = np.roll(camarilla_H3, 1)
    camarilla_L3 = np.roll(camarilla_L3, 1)
    camarilla_H4 = np.roll(camarilla_H4, 1)
    camarilla_L4 = np.roll(camarilla_L4, 1)
    # First day has no previous day
    camarilla_H3[0] = np.nan
    camarilla_L3[0] = np.nan
    camarilla_H4[0] = np.nan
    camarilla_L4[0] = np.nan
    
    # Calculate 1d volatility (ATR as percentage of price)
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    volatility = atr_1d / close_1d  # ATR as fraction of price
    
    # Align 1d indicators to 4h timeframe
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    volatility_aligned = align_htf_to_ltf(prices, df_1d, volatility)
    
    # 4h data for entry execution and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss (4h)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, 2) + 1  # Need at least 2 days for volatility
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(volatility_aligned[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation: volume above average
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Volatility filter: avoid extremely volatile markets (chaotic)
        vol_ok = volatility_aligned[i] < VOLATILITY_THRESHOLD
        
        # Camarilla level conditions
        at_H3 = abs(close[i] - camarilla_H3_aligned[i]) < (camarilla_H4_aligned[i] - camarilla_H3_aligned[i]) * 0.1
        at_L3 = abs(close[i] - camarilla_L3_aligned[i]) < (camarilla_L3_aligned[i] - camarilla_L4_aligned[i]) * 0.1
        
        # Generate signals
        if position == 0:
            # Long at L3 support with volume and low volatility
            if at_L3 and volume_ok and vol_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short at H3 resistance with volume and low volatility
            elif at_H3 and volume_ok and vol_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long at H3 (opposite level) or stop
            if close[i] >= camarilla_H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short at L3 (opposite level) or stop
            if close[i] <= camarilla_L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals