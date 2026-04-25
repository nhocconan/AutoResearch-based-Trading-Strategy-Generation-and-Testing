#!/usr/bin/env python3
"""
6h Volume-Weighted RSI with 12h Trend Filter and ATR Trailing Stop
Hypothesis: Volume-weighted RSI(14) on 6h filters noise by emphasizing price moves on high volume,
providing more reliable overbought/oversold signals. Combined with 12h EMA50 trend filter,
this strategy works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).
ATR-based trailing stop manages risk. Target: 50-150 trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vw_rsi(close, volume, period=14):
    """Calculate Volume-Weighted RSI"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan)
    
    # Price changes
    delta = np.diff(close, prepend=close[0])
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted gains and losses
    vol_gains = gains * volume
    vol_losses = losses * volume
    
    # Smoothed averages using Wilder's smoothing (alpha = 1/period)
    avg_vol_gain = np.zeros_like(close)
    avg_vol_loss = np.zeros_like(close)
    
    # Initialize first values
    avg_vol_gain[period] = np.mean(vol_gains[1:period+1])
    avg_vol_loss[period] = np.mean(vol_losses[1:period+1])
    
    # Wilder's smoothing
    for i in range(period + 1, len(close)):
        avg_vol_gain[i] = (avg_vol_gain[i-1] * (period - 1) + vol_gains[i]) / period
        avg_vol_loss[i] = (avg_vol_loss[i-1] * (period - 1) + vol_losses[i]) / period
    
    # Calculate RSI
    rs = np.divide(avg_vol_gain, avg_vol_loss, out=np.full_like(avg_vol_gain, np.nan), where=avg_vol_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA50 trend filter (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = calculate_ema(df_12h['close'].values, 50)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume-weighted RSI(14) on 6h
    vw_rsi = calculate_vw_rsi(close, volume, 14)
    
    # ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start index: need enough for VW RSI and ATR
    start_idx = max(14, 14) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vw_rsi[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # VW RSI conditions
        oversold = vw_rsi[i] < 30
        overbought = vw_rsi[i] > 70
        
        if position == 0:
            # Look for entry signals - require: VW RSI extreme + 12h EMA50 trend alignment
            long_entry = oversold and (curr_close > ema_50_12h_aligned[i])
            short_entry = overbought and (curr_close < ema_50_12h_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high_since_entry = curr_high
                lowest_low_since_entry = curr_low
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_high_since_entry = curr_high
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: update highest high and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: VW RSI returns to neutral, trend change, or ATR trailing stop
            trailing_stop = highest_high_since_entry - 2.5 * atr[i]
            if vw_rsi[i] > 50 or curr_close < ema_50_12h_aligned[i] or curr_close < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest low and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: VW RSI returns to neutral, trend change, or ATR trailing stop
            trailing_stop = lowest_low_since_entry + 2.5 * atr[i]
            if vw_rsi[i] < 50 or curr_close > ema_50_12h_aligned[i] or curr_close > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VolumeWeightedRSI_12hEMA50_Trend_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0