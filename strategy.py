#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Camarilla pivot levels for breakout and fade signals.
# Long when price breaks above R4 with volume confirmation and price > 1w EMA200.
# Short when price breaks below S4 with volume confirmation and price < 1w EMA200.
# Fade longs at R3 and shorts at S3 with reduced position size.
# Uses 1-week EMA200 as trend filter to align with higher timeframe trend.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull (R4 breakouts) and bear (S4 breakdowns) markets.
# Camarilla levels provide mathematically derived support/resistance that adapt to volatility.

name = "exp_13787_6h_camarilla1d_ema200w_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC
EMA_TREND_PERIOD = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
FADE_SIZE = 0.125  # Half position for fade trades
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Camarilla formula based on previous period's range
    range_ = high - low
    # Calculate levels
    r4 = close + range_ * 1.1 / 2
    r3 = close + range_ * 1.1 / 4
    s3 = close - range_ * 1.1 / 4
    s4 = close - range_ * 1.1 / 2
    return r4, r3, s3, s4

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
    
    # Load 1d data for Camarilla pivot levels (from previous day) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # Set first value to NaN as there's no previous day
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    camarilla_r4, camarilla_r3, camarilla_s3, camarilla_s4 = calculate_camarilla(
        high_1d_prev, low_1d_prev, close_1d_prev
    )
    
    # Load 1w data for EMA200 trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200w = calculate_ema(close_1w, EMA_TREND_PERIOD)
    
    # Align indicators to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_200w_aligned = align_htf_to_ltf(prices, df_1w, ema_200w)
    
    # 6h data for entry timing and ATR
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
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_200w_aligned[i]) or np.isnan(volume_ma[i])):
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
        
        # Trend direction from 1w EMA200
        above_ema = close[i] > ema_200w_aligned[i]
        below_ema = close[i] < ema_200w_aligned[i]
        
        # Breakout signals
        long_breakout = volume_ok and above_ema and close[i] > camarilla_r4_aligned[i]
        short_breakout = volume_ok and below_ema and close[i] < camarilla_s4_aligned[i]
        
        # Fade signals (counter-trend at R3/S3)
        long_fade = volume_ok and below_ema and close[i] < camarilla_r3_aligned[i] and close[i] > camarilla_s3_aligned[i]
        short_fade = volume_ok and above_ema and close[i] > camarilla_s3_aligned[i] and close[i] < camarilla_r3_aligned[i]
        
        # Generate signals
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif long_fade:
                signals[i] = FADE_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_fade:
                signals[i] = -FADE_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on close below S3 (support break) or above R4 with weakening momentum
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE if position == 1 else 0.0
        elif position == -1:
            # Exit short on close above R3 (resistance break) or below S4 with weakening momentum
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE if position == -1 else 0.0
    
    return signals