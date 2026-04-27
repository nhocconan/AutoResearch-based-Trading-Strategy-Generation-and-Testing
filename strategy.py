#!/usr/bin/env python3
"""
6h_RSI2_MeanReversion_1dTrendFilter_v1
Hypothesis: On 6h timeframe, use 2-period RSI for mean reversion entries (long when RSI2<10, short when RSI2>90) 
only in direction of 1d EMA50 trend. Volume spike confirms institutional participation. 
Designed for low trade frequency (15-25/year) to minimize fee drag while capturing mean reversion 
in both bull and bear markets by aligning with higher timeframe trend.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 2-period RSI on 6h
    if len(close) < 3:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi2 = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for RSI2 and volume avg
    start_idx = max(2, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi2[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        rsi_val = rsi2[i]
        ema_trend = ema_1d_aligned[i]
        size = 0.25  # 25% position size to manage risk
        
        if position == 0:
            # Flat - look for mean reversion in direction of 1d trend with volume confirmation
            # Long: RSI2 < 10 (oversold) AND price above 1d EMA50 AND volume spike
            long_entry = (rsi_val < 10) and (close_val > ema_trend) and volume_spike[i]
            # Short: RSI2 > 90 (overbought) AND price below 1d EMA50 AND volume spike
            short_entry = (rsi_val > 90) and (close_val < ema_trend) and volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when RSI2 crosses above 50 (mean reversion complete) or opposite signal
            exit_condition = (rsi_val > 50) or \
                           ((rsi_val > 90) and (close_val < ema_trend) and volume_spike[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when RSI2 crosses below 50 (mean reversion complete) or opposite signal
            exit_condition = (rsi_val < 50) or \
                           ((rsi_val < 10) and (close_val > ema_trend) and volume_spike[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI2_MeanReversion_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0