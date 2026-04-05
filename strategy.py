#!/usr/bin/env python3
"""
Experiment #11379: 6h Camarilla Pivot + Volume Spike + 12h Trend Filter
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) work well on 6h timeframe.
In ranging markets: fade extreme touches of R3/S3 with volume confirmation.
In trending markets: breakout continuation when price closes beyond R4/S4 with 12h trend alignment.
Volume filter ensures institutional participation. Designed for both bull (breakouts) and bear (mean reversion at extremes).
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11379_6h_camarilla_pivot_vol_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
TREND_HTF = '12h'

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price for the period
    typical_price = (high + low + close) / 3
    range_val = high - low
    
    # Camarilla levels
    R4 = close + range_val * 1.1 / 2
    R3 = close + range_val * 1.1 / 4
    S3 = close - range_val * 1.1 / 4
    S4 = close - range_val * 1.1 / 2
    
    return R4, R3, S3, S4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
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
    
    # Load daily data for Camarilla (using previous day's data)
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    camarilla_R4, camarilla_R3, camarilla_S3, camarilla_S4 = calculate_camarilla(high_daily, low_daily, close_daily)
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 day to avoid look-ahead)
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_R4)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_S3)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_S4)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, TREND_HTF)
    ema_12h = calculate_ema(df_12h['close'].values, 21)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if data not available
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(camarilla_R4_aligned[i]) or np.isnan(camarilla_S4_aligned[i]) or
            np.isnan(ema_12h_aligned[i])):
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
        
        # Trend filter (12h)
        uptrend_12h = close[i] > ema_12h_aligned[i]
        downtrend_12h = close[i] < ema_12h_aligned[i]
        
        # Market regime detection based on price action
        # Determine if we're in ranging or trending market based on recent price action
        # Simple approach: if price is between S3 and R3, we consider it ranging
        # If price breaks beyond S4 or R4, we consider it trending
        price_in_middle = (camarilla_S3_aligned[i] < close[i] < camarilla_R3_aligned[i])
        price_beyond_R4 = close[i] > camarilla_R4_aligned[i]
        price_beyond_S4 = close[i] < camarilla_S4_aligned[i]
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        if price_in_middle and volume_ok:
            # Ranging market: mean reversion at extremes
            # Long near S3, Short near R3
            long_entry = (low[i] <= camarilla_S3_aligned[i] * 1.002) and (close[i] > camarilla_S3_aligned[i])
            short_entry = (high[i] >= camarilla_R3_aligned[i] * 0.998) and (close[i] < camarilla_R3_aligned[i])
        elif (price_beyond_R4 or price_beyond_S4) and volume_ok:
            # Trending market: breakout continuation
            if price_beyond_R4 and uptrend_12h:
                long_entry = True
            elif price_beyond_S4 and downtrend_12h:
                short_entry = True
        
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