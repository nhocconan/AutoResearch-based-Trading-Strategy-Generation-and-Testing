#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with 1d RSI filter and volume confirmation.
# Long when KAMA rising (trending up), 1d RSI < 50 (avoid overbought), and volume > 1.5x 20-period average.
# Short when KAMA falling (trending down), 1d RSI > 50 (avoid oversold), and volume > 1.5x 20-period average.
# Exit when KAMA direction changes.
# Uses KAMA for adaptive trend following, RSI to avoid extremes, volume to confirm strength.
# Target: 50-120 total trades over 4 years (12-30/year) for low fee drag.

name = "12h_KAMA_1dRSI_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h KAMA (adaptive moving average)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will fix below
    
    # Proper ER calculation
    price_change = np.abs(np.subtract(close[10:], close[:-10]))
    volatility_sum = np.array([np.sum(np.abs(np.diff(close[i:i+10]))) for i in range(len(close)-9)])
    volatility_sum = np.concatenate([np.full(9, np.nan), volatility_sum])
    er = np.divide(price_change, volatility_sum, out=np.zeros_like(price_change), where=volatility_sum!=0)
    er = np.concatenate([np.full(10, np.nan), er])
    
    # Smoothing constants
    fast_sc = np.power(2 / (2 + 1), 2)  # SC for EMA(2)
    slow_sc = np.power(2 / (30 + 1), 2)  # SC for EMA(30)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at index 9
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction: rising if current > previous, falling if current < previous
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    # 12h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma20[:10] = np.nan
    vol_ma20[-10:] = np.nan
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI (14-period) on 1d data
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Average gain and loss
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    # Avoid division by zero
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100  # when no losses, RSI=100
    rsi[avg_gain == 0] = 0    # when no gains, RSI=0
    
    # Align 1d RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(kama_rising[i]) or np.isnan(kama_falling[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA rising, RSI < 50 (not overbought), volume spike
            long_cond = kama_rising[i] and (rsi_aligned[i] < 50) and volume_filter[i]
            # Short conditions: KAMA falling, RSI > 50 (not oversold), volume spike
            short_cond = kama_falling[i] and (rsi_aligned[i] > 50) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA starts falling
            if kama_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA starts rising
            if kama_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals