#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(2) mean reversion + volume spike confirmation
# Long when KAMA direction is up (trending up) AND RSI(2) < 10 (oversold) AND volume > 1.5x 20-day average
# Short when KAMA direction is down (trending down) AND RSI(2) > 90 (overbought) AND volume > 1.5x 20-day average
# Exit when RSI(2) crosses back to neutral (40-60 range) OR KAMA direction flips
# KAMA adapts to market noise, reducing whipsaws in choppy markets
# RSI(2) captures short-term mean reversion extremes
# Volume spike confirms institutional participation at turning points
# Target: 7-25 trades/year per symbol (30-100 total over 4 years) for 1d timeframe
# Discrete sizing (0.25) to limit fee drag

name = "1d_KAMA_Trend_RSI2_MeanRev_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for HTF trend context (optional filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(data, period=10, fast=2, slow=30):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        # Efficiency Ratio
        change = np.abs(np.diff(data, n=period))
        volatility = np.sum(np.abs(np.diff(data)), axis=0) if len(data) > 1 else 0
        er = np.zeros_like(data)
        er[period:] = change[period-1:] / np.maximum(volatility[period-1:], 1e-10)
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama_vals = np.full_like(data, np.nan, dtype=float)
        kama_vals[period-1] = data[period-1]
        for i in range(period, len(data)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (data[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, period=10, fast=2, slow=30)
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.zeros_like(kama_vals)
    kama_dir[1:] = np.where(kama_vals[1:] > kama_vals[:-1], 1, -1)
    
    # RSI(2) for short-term mean reversion
    def rsi(data, period=2):
        if len(data) < period + 1:
            return np.full_like(data, 50.0, dtype=float)
        deltas = np.diff(data)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = np.where(down == 0, np.inf, up / down)
        rsi_vals = np.full_like(data, 100. - (100. / (1. + rs)), dtype=float)
        for i in range(period+1, len(data)):
            delta = deltas[i-1]
            upval = max(delta, 0)
            downval = max(-delta, 0)
            up = (up * (period-1) + upval) / period
            down = (down * (period-1) + downval) / period
            rs = np.where(down == 0, np.inf, up / down)
            rsi_vals[i] = 100. - (100. / (1. + rs))
        return rsi_vals
    
    rsi_2 = rsi(close, period=2)
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(kama_vals[i]) or 
            np.isnan(rsi_2[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA up AND RSI(2) oversold AND volume spike
            if (kama_dir[i] == 1 and 
                rsi_2[i] < 10 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA down AND RSI(2) overbought AND volume spike
            elif (kama_dir[i] == -1 and 
                  rsi_2[i] > 90 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI(2) crosses above 40 (mean reversion unwind) OR KAMA flips down
            if (rsi_2[i] > 40 or 
                kama_dir[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI(2) crosses below 60 (mean reversion unwind) OR KAMA flips up
            if (rsi_2[i] < 60 or 
                kama_dir[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals