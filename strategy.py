#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Camarilla pivot levels with volume confirmation.
# Goes long when price bounces off pivot support (S1/S2) with above-average volume,
# short when price rejects at pivot resistance (R1/R2) with volume.
# Uses 1d EMA200 as trend filter to avoid counter-trend trades.
# Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Camarilla levels provide clear support/resistance levels that work in ranging markets.
# Works in bull (bounce off support) and bear (rejection at resistance) markets.

name = "exp_13798_4h_camarilla1d_ema200_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1
EMA_TREND_PERIOD = 200
VOLUME_MA_PERIOD = 10
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    # Camarilla levels
    s1 = close - (range_ * 1.1 / 12)
    s2 = close - (range_ * 1.1 / 6)
    r1 = close + (range_ * 1.1 / 12)
    r2 = close + (range_ * 1.1 / 6)
    return s1, s2, r1, r2

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and EMA trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_s1, camarilla_s2, camarilla_r1, camarilla_r2 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate 1d EMA200 for trend filter
    ema_1d = calculate_ema(close_1d, EMA_TREND_PERIOD)
    
    # Align 1d indicators to 4h timeframe
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data for entry timing and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_TREND_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_s2_aligned[i]) or \
           np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_r2_aligned[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 1d EMA200
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Camarilla bounce/rejection signals
        # Long: price near support (S1/S2) with bullish bias
        near_support = (close[i] <= camarilla_s1_aligned[i] * 1.005) or (close[i] <= camarilla_s2_aligned[i] * 1.005)
        long_signal = volume_ok and near_support and above_ema
        
        # Short: price near resistance (R1/R2) with bearish bias
        near_resistance = (close[i] >= camarilla_r1_aligned[i] * 0.995) or (close[i] >= camarilla_r2_aligned[i] * 0.995)
        short_signal = volume_ok and near_resistance and below_ema
        
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
            # Exit long on close below S2 (support broken)
            if close[i] < camarilla_s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above R2 (resistance broken)
            if close[i] > camarilla_r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals