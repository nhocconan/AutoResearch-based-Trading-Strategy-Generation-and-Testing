#!/usr/bin/env python3
"""
12h Camarilla pivot reversal with 1d trend filter and volume confirmation.
Hypothesis: Price reverses from Camarilla levels (S3/S4 for long, R3/R4 for short)
when aligned with daily trend and volume confirmation. Works in ranging markets
and avoids false breakouts by using mean-reversion at extremes. Target: 50-150 trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14265_12h_camarilla_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous 12h bar
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    
    # Avoid look-ahead: use previous bar's data
    prev_range = prev_high - prev_low
    r4 = prev_close + (prev_range * 1.1 / 2)
    r3 = prev_close + (prev_range * 1.1 / 4)
    s3 = prev_close - (prev_range * 1.1 / 4)
    s4 = prev_close - (prev_range * 1.1 / 2)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for volume, 14 for ATR)
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(r4[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(s4[i]):
            if position != 0:
                signals[i] = position * 0.25
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
        
        # Camarilla reversal signals with 1d trend filter and volume
        # Long: price touches S3/S4 + price > 1d EMA + volume
        # Short: price touches R3/R4 + price < 1d EMA + volume
        touch_s3_s4 = (low[i] <= s3[i]) or (low[i] <= s4[i])
        touch_r3_r4 = (high[i] >= r3[i]) or (high[i] >= r4[i])
        
        long_signal = touch_s3_s4 and (close[i] > ema_1d_aligned[i]) and vol_filter[i]
        short_signal = touch_r3_r4 and (close[i] < ema_1d_aligned[i]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or when price reaches Camarilla pivot (mean reversion target)
            if close[i] <= stop_price or close[i] >= (prev_close[i-1] + (prev_range[i-1] * 1.1 / 6)):  # R3 level
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or when price reaches Camarilla pivot (mean reversion target)
            if close[i] >= stop_price or close[i] <= (prev_close[i-1] - (prev_range[i-1] * 1.1 / 6)):  # S3 level
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals