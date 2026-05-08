#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h RSI(14) and 1d EMA200 for trend direction, with volume confirmation.
# Long when 4h RSI < 30 (oversold), 1d EMA200 rising, and volume > 1.5x 20-period average.
# Short when 4h RSI > 70 (overbought), 1d EMA200 falling, and volume > 1.5x 20-period average.
# Exit when RSI crosses back above 50 (long) or below 50 (short).
# RSI provides mean-reversion signals, EMA200 filters trend, volume confirms strength.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.

name = "1h_RSI_4h_EMA200_1d_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for RSI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate RSI on 4h close
    delta = pd.Series(df_4h['close']).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_4h = align_htf_to_ltf(prices, df_4h, rsi)
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # EMA200 direction
    ema200_rising = np.zeros_like(ema200_1d_aligned, dtype=bool)
    ema200_falling = np.zeros_like(ema200_1d_aligned, dtype=bool)
    ema200_rising[1:] = ema200_1d_aligned[1:] > ema200_1d_aligned[:-1]
    ema200_falling[1:] = ema200_1d_aligned[1:] < ema200_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 14, 2)  # Sufficient warmup for EMA200 and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_4h[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(ema200_rising[i]) or np.isnan(ema200_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI oversold, EMA200 rising, volume filter
            long_cond = (rsi_4h[i] < 30) and ema200_rising[i] and volume_filter[i]
            # Short conditions: RSI overbought, EMA200 falling, volume filter
            short_cond = (rsi_4h[i] > 70) and ema200_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI crosses back above 50
            if rsi_4h[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI crosses back below 50
            if rsi_4h[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals