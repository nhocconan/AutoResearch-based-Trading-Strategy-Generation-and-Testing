#!/usr/bin/env python3
"""
Experiment #8835: 6h Camarilla pivot + volume confirmation + weekly trend filter
Hypothesis: Camarilla pivot levels from daily timeframe provide institutional support/resistance.
Fade at R3/S3 (mean reversion in range), breakout continuation at R4/S4 (trend following).
Weekly trend filter ensures alignment with higher timeframe momentum. Volume confirmation
filters for institutional participation. Targets 50-150 trades over 4 years.
Works in both bull/bear: mean reversion in ranges, trend following in breaks.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8835_6h_camarilla_pivot_vol_wtrend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Daily pivot
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
TREND_PERIOD = 50  # Weekly EMA for trend filter
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

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=TREND_PERIOD, adjust=False, min_periods=TREND_PERIOD).mean().values
    
    # Price relative to 1w EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1w > ema_1w, 1, 
                     np.where(close_1w < ema_1w, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1w, price_vs_ema)
    
    # Calculate Camarilla levels from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: 
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.125 * (High - Low)
    # L3 = Close - 1.125 * (High - Low)
    # H2 = Close + 0.75 * (High - Low)
    # L2 = Close - 0.75 * (High - Low)
    # H1 = Close + 0.5 * (High - Low)
    # L1 = Close - 0.5 * (High - Low)
    pivot_range = high_1d - low_1d
    r4 = close_1d + 1.5 * pivot_range
    r3 = close_1d + 1.125 * pivot_range
    s3 = close_1d - 1.125 * pivot_range
    s4 = close_1d - 1.5 * pivot_range
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine market bias from 1w EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1w price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1w price below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Camarilla-based entry logic
        # Fade at R3/S3 (mean reversion), breakout at R4/S4 (continuation)
        fade_long = (not bull_bias) and close[i] <= r3_aligned[i] and close[i] > s3_aligned[i] and volume_confirmed
        fade_short = (not bear_bias) and close[i] >= s3_aligned[i] and close[i] < r3_aligned[i] and volume_confirmed
        breakout_long = bull_bias and close[i] > r4_aligned[i-1] and volume_confirmed
        breakout_short = bear_bias and close[i] < s4_aligned[i-1] and volume_confirmed
        
        # Entry conditions
        long_entry = fade_long or breakout_long
        short_entry = fade_short or breakout_short
        
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