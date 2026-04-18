#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h RSI trend filter and 1d volume regime filter.
- Long: RSI(14) > 50 on 4h (bullish momentum) AND volume > 1.5x 20-period average on 1h
- Short: RSI(14) < 50 on 4h (bearish momentum) AND volume > 1.5x 20-period average on 1h
- Uses 4h for trend direction (reduces whipsaw), 1h for entry timing with volume confirmation
- Position size: 0.20 (20% of capital) to manage drawdown
- Session filter: 08-20 UTC to avoid low-liquidity hours
Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for RSI trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate RSI(14) on 4h
    def calculate_rsi(prices, period=14):
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(len(prices), np.nan)
        avg_loss = np.full(len(prices), np.nan)
        
        avg_gain[period] = np.nanmean(gain[:period])
        avg_loss[period] = np.nanmean(loss[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_14_4h = calculate_rsi(close_4h, 14)
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # Calculate 20-period volume moving average on 1h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need RSI (14+1) and volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session or missing data
        if not in_session[i] or np.isnan(rsi_14_4h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: bullish 4h RSI (>50) + volume confirmation
            if rsi_14_4h_aligned[i] > 50 and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: bearish 4h RSI (<50) + volume confirmation
            elif rsi_14_4h_aligned[i] < 50 and vol_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: bearish 4h RSI (<50) or loss of volume confirmation
            if rsi_14_4h_aligned[i] < 50 or not vol_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: bullish 4h RSI (>50) or loss of volume confirmation
            if rsi_14_4h_aligned[i] > 50 or not vol_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI4h_Volume"
timeframe = "1h"
leverage = 1.0