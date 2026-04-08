#!/usr/bin/env python3
# 4h_rsi_ema_crossover_1d_trend_volume_v1
# Hypothesis: 4h RSI crossing above/below EMA(21) with 1d EMA(50) trend filter and volume confirmation.
# RSI > EMA indicates bullish momentum; RSI < EMA indicates bearish momentum.
# Volume confirms participation. Trend filter avoids counter-trend trades.
# Target: 20-40 trades/year with position size 0.25 to minimize fee drag.

name = "4h_rsi_ema_crossover_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # EMA(21) on RSI
    rsi_ema = pd.Series(rsi).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1d EMA trend filter (50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(21, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(rsi_ema[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if RSI crosses below EMA or trend fails
            if rsi[i] < rsi_ema[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if RSI crosses above EMA or trend fails
            if rsi[i] > rsi_ema[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI above EMA with uptrend and volume
            if rsi[i] > rsi_ema[i] and close[i] > ema_1d_aligned[i] and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI below EMA with downtrend and volume
            elif rsi[i] < rsi_ema[i] and close[i] < ema_1d_aligned[i] and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals