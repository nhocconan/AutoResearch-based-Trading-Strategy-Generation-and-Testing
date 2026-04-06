#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal strategy with 1d trend filter.
# Camarilla levels from prior day: long near S3/S4 (strong support), short near R3/R4 (strong resistance).
# Uses 1d EMA200 for trend filter to avoid counter-trend trades.
# Works in bull/bear because reversals at strong levels occur in all regimes.
# Target: 80-180 trades over 4 years (20-45/year) to balance opportunity and cost.

name = "exp_13051_6h_camarilla_reversal_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard multiplier for Camarilla
EMA_TREND_PERIOD = 200
VOLUME_MA_PERIOD = 24
VOLUME_THRESHOLD = 1.3
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_TREND_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate prior day's Camarilla levels (using prior day's OHLC)
    # We need to shift OHLC by 1 to get *completed* day's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's range
    range_1d = high_1d - low_1d
    
    # Camarilla levels for prior day
    # R4 = close + range * 1.1/2
    # R3 = close + range * 1.1/4
    # S3 = close - range * 1.1/4
    # S4 = close - range * 1.1/2
    camarilla_r4 = close_1d + range_1d * CAMARILLA_MULT / 2
    camarilla_r3 = close_1d + range_1d * CAMARILLA_MULT / 4
    camarilla_s3 = close_1d - range_1d * CAMARILLA_MULT / 4
    camarilla_s4 = close_1d - range_1d * CAMARILLA_MULT / 2
    
    # Align Camarilla levels to 6h timeframe (using prior day's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1d_aligned[i]):
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
        
        # Trend filter: price above/below daily EMA200
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Reversal signals at Camarilla levels
        # Long near S3/S4 in uptrend
        long_signal = (volume_ok and uptrend and 
                      (low[i] <= camarilla_s3_aligned[i] or low[i] <= camarilla_s4_aligned[i]) and
                      close[i] > camarilla_s3_aligned[i])  # Confirm reversal
        
        # Short near R3/R4 in downtrend
        short_signal = (volume_ok and downtrend and 
                       (high[i] >= camarilla_r3_aligned[i] or high[i] >= camarilla_r4_aligned[i]) and
                       close[i] < camarilla_r3_aligned[i])  # Confirm reversal
        
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