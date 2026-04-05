#!/usr/bin/env python3
"""
Experiment #7667: 6-hour Camarilla pivot + volume confirmation with 1-week trend filter.
Hypothesis: Camarilla levels derived from previous day's range act as strong support/resistance.
In bull markets (price > 1w EMA50), buy near S3/S4 with volume confirmation.
In bear markets (price < 1w EMA50), sell near R3/R4 with volume confirmation.
Uses 1-day OHLC to calculate Camarilla levels for the 6h chart.
Targets 75-150 trades over 4 years (19-38/year) with precise level-based entries.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7667_6h_camarilla_1w_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_LOOKBACK = 1  # use previous day's OHLC
EMA_TREND = 50
VOLUME_MA_PERIOD = 24  # 4 days of 6h data
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla levels from previous day's OHLC"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close
    c = close
    h = high
    l = low
    # Camarilla levels
    r4 = c + (range_val * 1.1 / 2)
    r3 = c + (range_val * 1.1 / 4)
    r2 = c + (range_val * 1.1 / 6)
    r1 = c + (range_val * 1.1 / 12)
    s1 = c - (range_val * 1.1 / 12)
    s2 = c - (range_val * 1.1 / 6)
    s3 = c - (range_val * 1.1 / 4)
    s4 = c - (range_val * 1.1 / 2)
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize Camarilla arrays
    r1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    r4 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    
    # Calculate Camarilla for each day, propagate to intraday periods
    for i in range(len(df_1d)):
        r1_val, r2_val, r3_val, r4_val, s1_val, s2_val, s3_val, s4_val = calculate_camarilla(
            high_1d[i], low_1d[i], close_1d[i]
        )
        # Find corresponding 6h periods for this day
        # Each day has 4 six-hour periods (00:00-06:00, 06:00-12:00, 12:00-18:00, 18:00-00:00)
        start_idx = i * 4
        end_idx = start_idx + 4
        if end_idx > n:
            end_idx = n
        if start_idx < n:
            r1[start_idx:end_idx] = r1_val
            r2[start_idx:end_idx] = r2_val
            r3[start_idx:end_idx] = r3_val
            r4[start_idx:end_idx] = r4_val
            s1[start_idx:end_idx] = s1_val
            s2[start_idx:end_idx] = s2_val
            s3[start_idx:end_idx] = s3_val
            s4[start_idx:end_idx] = s4_val
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_TREND, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_50_aligned[i]):
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
        
        # Determine market regime
        bull_regime = close[i] > ema_1w_50_aligned[i]   # price above 1w EMA50
        bear_regime = close[i] < ema_1w_50_aligned[i]   # price below 1w EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions based on Camarilla levels
        # Long near support (S3/S4) in bull market, short near resistance (R3/R4) in bear market
        long_entry = bull_regime and volume_confirmed and (
            (low[i] <= s3[i] and not np.isnan(s3[i])) or 
            (low[i] <= s4[i] and not np.isnan(s4[i]))
        )
        short_entry = bear_regime and volume_confirmed and (
            (high[i] >= r3[i] and not np.isnan(r3[i])) or 
            (high[i] >= r4[i] and not np.isnan(r4[i]))
        )
        
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