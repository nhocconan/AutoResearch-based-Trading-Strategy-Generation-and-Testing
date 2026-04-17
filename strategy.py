#!/usr/bin/env python3
"""
Hypothesis: 6h Volume-Weighted RSI + 12h Trend Filter.
Long when VW_RSI(14) < 30 and 12h EMA50 > EMA200 (uptrend).
Short when VW_RSI(14) > 70 and 12h EMA50 < EMA200 (downtrend).
Exit when VW_RSI crosses 50 opposite direction or trend reverses.
Uses 6h for VW_RSI calculation (volume confirmation), 12h for trend filter.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 and EMA200
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h indicators
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Calculate 6h Volume-Weighted RSI
    def calculate_vw_rsi(close, volume, period=14):
        # Typical price
        tp = (high + low + close) / 3.0
        # Volume-weighted typical price change
        vwtp = tp * volume
        
        # Calculate changes
        change = np.diff(vwtp, prepend=vwtp[0])
        
        # Separate gains and losses
        gains = np.where(change > 0, change, 0.0)
        losses = np.where(change < 0, -change, 0.0)
        
        # Wilder's smoothing (EMA with alpha=1/period)
        avg_gain = np.zeros_like(gains)
        avg_loss = np.zeros_like(losses)
        
        # First average: simple mean
        if len(gains) > period:
            avg_gain[period] = np.mean(gains[1:period+1])
            avg_loss[period] = np.mean(losses[1:period+1])
            
            # Subsequent: Wilder's smoothing
            for i in range(period+1, len(gains)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period
        
        # Avoid division by zero
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi
    
    # Calculate VW_RSI
    vw_rsi = calculate_vw_rsi(close, volume, 14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vw_rsi[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(ema200_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend determination from 12h
        ema50 = ema50_12h_aligned[i]
        ema200 = ema200_12h_aligned[i]
        is_uptrend = ema50 > ema200
        is_downtrend = ema50 < ema200
        
        # VW_RSI signals
        rsi = vw_rsi[i]
        oversold = rsi < 30
        overbought = rsi > 70
        exit_long = rsi > 50
        exit_short = rsi < 50
        
        if position == 0:
            # Long: Oversold + Uptrend
            if oversold and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Overbought + Downtrend
            elif overbought and is_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses above 50 OR trend turns down
            if exit_long or is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50 OR trend turns up
            if exit_short or is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VWRSI_12hEMATrend"
timeframe = "6h"
leverage = 1.0