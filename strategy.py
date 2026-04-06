#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot reversal strategy with 1-day trend filter and volume confirmation.
# Uses 1-day Camarilla levels (R3/S3 for reversal, R4/S4 for breakout) with 1-day EMA trend filter.
# In uptrend: buy at S3 bounce, sell at R3; in downtrend: sell at R3 bounce, buy at S3.
# Volume confirms institutional participation. Target: 75-150 total trades over 4 years.
# Works in bull markets (trend-following breaks) and bear markets (mean-reversion at pivot levels).
# Camarilla provides precise support/resistance levels that work well in ranging and trending markets.

name = "exp_13691_6h_camarilla1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC for Camarilla calculation
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Camarilla formula based on previous period's OHLC
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    # Resistance levels
    r4 = close + range_hl * 1.1 / 2
    r3 = close + range_hl * 1.1 / 4
    r2 = close + range_hl * 1.1 / 6
    r1 = close + range_hl * 1.1 / 12
    
    # Support levels
    s1 = close - range_hl * 1.1 / 12
    s2 = close - range_hl * 1.1 / 6
    s3 = close - range_hl * 1.1 / 4
    s4 = close - range_hl * 1.1 / 2
    
    return {
        'r4': r4, 'r3': r3, 'r2': r2, 'r1': r1,
        's1': s1, 's2': s2, 's3': s3, 's4': s4,
        'pivot': pivot
    }

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla levels and trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_for_cama = df_1d['close'].values
    
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    
    # Calculate Camarilla for each day (using previous day's OHLC)
    for i in range(1, len(df_1d)):
        camarilla = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d_for_cama[i-1])
        camarilla_r3[i] = camarilla['r3']
        camarilla_s3[i] = camarilla['s3']
        camarilla_r4[i] = camarilla['r4']
        camarilla_s4[i] = camarilla['s4']
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma[i])):
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
        
        # Trend direction from 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Camarilla levels
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # Initialize signal
        signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
        
        # Trading logic
        if position == 0:
            # Look for reversal signals at S3/R3 with volume confirmation
            if volume_ok:
                # Long setup: price bounces off S3 in uptrend or breaks above R4
                long_setup = (above_ema and close[i] > s3 and close[i-1] <= s3) or \
                            (close[i] > r4 and close[i-1] <= r4)
                
                # Short setup: price bounces off R3 in downtrend or breaks below S4
                short_setup = (below_ema and close[i] < r3 and close[i-1] >= r3) or \
                             (close[i] < s4 and close[i-1] >= s4)
                
                if long_setup:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif short_setup:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
        elif position == 1:
            # Exit long: stop loss, take profit at R3, or reversal at R3
            if close[i] >= r3:  # Take profit at R3 or stop loss
                signals[i] = 0.0
                position = 0
            elif below_ema and close[i] < r3 and close[i-1] >= r3:  # Reversal signal
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Exit short: stop loss, take profit at S3, or reversal at S3
            if close[i] <= s3:  # Take profit at S3 or stop loss
                signals[i] = 0.0
                position = 0
            elif above_ema and close[i] > s3 and close[i-1] <= s3:  # Reversal signal
                signals[i] = 0.0
                position = 0
    
    return signals