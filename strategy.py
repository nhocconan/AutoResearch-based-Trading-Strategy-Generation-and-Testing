#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with 1d RSI mean-reversion filter and volume confirmation.
# Long when KAMA slope > 0 (up trend), RSI(14) < 40 (oversold), and 1d volume > 1.5x 20-period average.
# Short when KAMA slope < 0 (down trend), RSI(14) > 60 (overbought), and 1d volume > 1.5x 20-period average.
# Exit when KAMA slope changes sign or volume filter fails.
# Uses 12h timeframe with 1d RSI and volume for higher timeframe context.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "12h_KAMA_RSI_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for RSI and volume
    df_d = get_htf_data(prices, '1d')
    if len(dfd) < 2:
        return np.zeros(n)
    
    # KAMA on 12h data
    def kama(close, er_period=10, fast=2, slow=30):
        change = np.abs(close - np.roll(close, er_period))
        volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_values = kama(close)
    kama_slope = np.diff(kama_values, prepend=kama_values[0])
    
    # Daily RSI(14)
    close_d = df_d['close'].values
    delta = np.diff(close_d, prepend=close_d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isnan(rsi)] = 50  # Neutral when undefined
    
    # Daily volume filter: current volume > 1.5x 20-period average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.5 * vol_ma20_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for KAMA and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_slope[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA up, RSI oversold, volume confirmation
            long_cond = (kama_slope[i] > 0) and (rsi_aligned[i] < 40) and volume_filter[i]
            # Short conditions: KAMA down, RSI overbought, volume confirmation
            short_cond = (kama_slope[i] < 0) and (rsi_aligned[i] > 60) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA turns down or volume filter fails
            if (kama_slope[i] <= 0) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA turns up or volume filter fails
            if (kama_slope[i] >= 0) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals