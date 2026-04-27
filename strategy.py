#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mplfinance import make_marketcolors
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA with 1d RSI filter and volume spike.
# Long when KAMA crosses above price (trend up) and 1d RSI > 50 (bullish bias) and volume > 2x average.
# Short when KAMA crosses below price (trend down) and 1d RSI < 50 (bearish bias) and volume > 2x average.
# Exit when KAMA crosses back through price (mean reversion).
# KAMA adapts to market noise, reducing whipsaw in ranging markets.
# Volume spike confirms institutional interest.
# Target: 15-35 trades per year on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI (14-period) for trend filter
    rsi_period = 14
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= rsi_period:
        avg_gain[rsi_period - 1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period - 1] = np.mean(loss[:rsi_period])
        for i in range(rsi_period, len(close_1d)):
            avg_gain[i] = (avg_gain[i - 1] * (rsi_period - 1) + gain[i - 1]) / rsi_period
            avg_loss[i] = (avg_loss[i - 1] * (rsi_period - 1) + loss[i - 1]) / rsi_period
    
    rs = np.full(len(close_1d), np.nan)
    rsi_1d = np.full(len(close_1d), np.nan)
    for i in range(rsi_period - 1, len(close_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi_1d[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi_1d[i] = 100
    
    # Calculate KAMA (10-period ER, 2/30 SC)
    kama_period = 10
    fast_sc = 2
    slow_sc = 30
    
    # Efficiency Ratio
    change = np.abs(np.subtract(close[kama_period:], close[:-kama_period]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly
    
    # Proper volatility calculation (sum of absolute changes)
    volatility = np.full(n, np.nan)
    for i in range(kama_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i - kama_period:i])))
    
    er = np.full(n, np.nan)
    for i in range(kama_period, n):
        if volatility[i] != 0:
            er[i] = change[i - kama_period] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing Constant
    sc = np.full(n, np.nan)
    for i in range(kama_period, n):
        sc[i] = (er[i] * (fast_sc / kama_period - slow_sc / kama_period) + slow_sc / kama_period) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    if len(close) >= kama_period:
        kama[kama_period - 1] = np.mean(close[:kama_period])
        for i in range(kama_period, n):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    # Align 1d RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume MA for spike detection (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need KAMA, RSI14, and volume MA20
    start_idx = max(kama_period, rsi_period, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume spike (>2x average)
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: KAMA crosses above price with 1d RSI > 50 and volume spike
            if kama[i] > price and kama[i - 1] <= price[i - 1] and \
               rsi_1d_aligned[i] > 50 and vol_filter:
                signals[i] = size
                position = 1
            # Short: KAMA crosses below price with 1d RSI < 50 and volume spike
            elif kama[i] < price and kama[i - 1] >= price[i - 1] and \
                 rsi_1d_aligned[i] < 50 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA crosses below price
            if kama[i] < price and kama[i - 1] >= price[i - 1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: KAMA crosses above price
            if kama[i] > price and kama[i - 1] <= price[i - 1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_KAMA10_RSI14_VolumeSpike"
timeframe = "12h"
leverage = 1.0