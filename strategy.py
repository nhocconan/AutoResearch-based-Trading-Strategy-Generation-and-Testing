#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation and ATR stoploss.
# Goes long when price touches Camarilla L3 with volume confirmation, short when touches H3.
# Uses 1-week trend filter (price above/below 200 EMA) to avoid counter-trend trades.
# Camarilla levels provide institutional support/resistance that work in both bull and bear markets.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13802_12h_camarilla1d_1w_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1
VOLUME_MA_PERIOD = 10
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
TREND_EMA_PERIOD = 200

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for intraday trading"""
    # Typical price for the period
    typical_price = (high + low + close) / 3
    # Range
    range_val = high - low
    # Camarilla levels
    h4 = close + range_val * 1.1 / 2
    h3 = close + range_val * 1.1 / 4
    h2 = close + range_val * 1.1 / 6
    h1 = close + range_val * 1.1 / 12
    l1 = close - range_val * 1.1 / 12
    l2 = close - range_val * 1.1 / 6
    l3 = close - range_val * 1.1 / 4
    l4 = close - range_val * 1.1 / 2
    return h4, h3, h2, h1, l1, l2, l3, l4

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
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    h4, h3, h2, h1, l1, l2, l3, l4 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Load weekly data for trend filter (200 EMA)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, TREND_EMA_PERIOD)
    
    # Align daily Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h2_aligned = align_htf_to_ltf(prices, df_1d, h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, l2)
    
    # Align weekly EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 12h data for entry timing and ATR
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
    start = max(20, VOLUME_MA_PERIOD) + 1  # Need enough data for calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma[i]):
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
        
        # Trend direction from weekly EMA
        above_weekly_ema = close[i] > ema_1w_aligned[i]
        below_weekly_ema = close[i] < ema_1w_aligned[i]
        
        # Camarilla level touches with volume confirmation
        # Long when price touches or goes below L3 with volume
        long_signal = volume_ok and above_weekly_ema and close[i] <= l3_aligned[i]
        # Short when price touches or goes above H3 with volume
        short_signal = volume_ok and below_weekly_ema and close[i] >= h3_aligned[i]
        
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
            # Exit long on touch of H3 or stop loss
            if close[i] >= h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on touch of L3 or stop loss
            if close[i] <= l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals