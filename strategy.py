#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot reversal with 1-day ATR filter and volume confirmation.
# In ranging markets, price tends to revert from Camarilla levels (H3/L3, H4/L4).
# The daily ATR ensures we only trade when volatility is sufficient to avoid chop.
# Volume confirmation filters out low-conviction moves.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# This strategy has shown promise on ETHUSDT and aims to work across BTC/ETH/SOL.

name = "exp_13285_12h_camarilla_pivot_atr_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_LOOKBACK = 1  # Use previous day for pivot calculation
ATR_PERIOD = 14
ATR_THRESHOLD_MULTIPLIER = 0.5  # Minimum ATR as % of price
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.3
SIGNAL_SIZE = 0.25

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given HLC"""
    range_ = high - low
    if range_ == 0:
        return close, close, close, close, close, close, close, close
    c = close + (range_ * 1.1 / 12)
    l3 = close - (range_ * 1.1 / 6)
    h3 = close + (range_ * 1.1 / 6)
    l4 = close - (range_ * 1.1 / 4)
    h4 = close + (range_ * 1.1 / 4)
    return l3, h3, l4, h4, c, c, c, c  # l3,h3,l4,h4,c5,c6,c7,c8 (we use l3,h3,l4,h4)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    atr_1d_avg = pd.Series(atr_1d).rolling(window=20, min_periods=1).mean().values  # 20-day avg ATR
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_avg)
    
    # Calculate daily Camarilla levels (using previous day's HLC)
    # We'll calculate these inside the loop using shifted values to avoid look-ahead
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    # Set first value to NaN since no previous day
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    close_1d_shift[0] = np.nan
    
    # Pre-calculate Camarilla levels for each day
    camarilla_data = []
    for i in range(len(high_1d)):
        if np.isnan(high_1d_shift[i]) or np.isnan(low_1d_shift[i]) or np.isnan(close_1d_shift[i]):
            camarilla_data.append((np.nan, np.nan, np.nan, np.nan))  # l3, h3, l4, h4
        else:
            l3, h3, l4, h4, _, _, _, _ = calculate_camarilla(high_1d_shift[i], low_1d_shift[i], close_1d_shift[i])
            camarilla_data.append((l3, h3, l4, h4))
    
    camarilla_data = np.array(camarilla_data)
    l3_1d = camarilla_data[:, 0]
    h3_1d = camarilla_data[:, 1]
    l4_1d = camarilla_data[:, 2]
    h4_1d = camarilla_data[:, 3]
    
    # Align Camarilla levels to 12h timeframe
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, ATR_PERIOD) + 1  # volume MA and ATR period
    
    for i in range(start, n):
        # Skip if Camarilla or ATR data not available
        if (np.isnan(l3_1d_aligned[i]) or np.isnan(h3_1d_aligned[i]) or 
            np.isnan(l4_1d_aligned[i]) or np.isnan(h4_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: ATR must be above threshold
        atr_threshold = close[i] * ATR_THRESHOLD_MULTIPLIER / 100  # ATR as % of price
        volatility_ok = atr_1d_aligned[i] > atr_threshold
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Price proximity to Camarilla levels (within 0.1% of level)
        proximity_threshold = 0.001  # 0.1%
        near_l3 = abs(close[i] - l3_1d_aligned[i]) / close[i] < proximity_threshold
        near_h3 = abs(close[i] - h3_1d_aligned[i]) / close[i] < proximity_threshold
        near_l4 = abs(close[i] - l4_1d_aligned[i]) / close[i] < proximity_threshold
        near_h4 = abs(close[i] - h4_1d_aligned[i]) / close[i] < proximity_threshold
        
        # Reversal logic: sell near resistance, buy near support
        sell_signal = (near_h3 or near_h4) and volatility_ok and volume_ok
        buy_signal = (near_l3 or near_l4) and volatility_ok and volume_ok
        
        # Generate signals
        if position == 0:
            if buy_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
            elif sell_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long if price reaches opposite resistance or volatility dries up
            if (near_h3 or near_h4) or not volatility_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short if price reaches opposite support or volatility dries up
            if (near_l3 or near_l4) or not volatility_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals