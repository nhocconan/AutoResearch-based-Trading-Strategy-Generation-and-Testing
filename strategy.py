#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Reversal_Momentum
Hypothesis: Combines Camarilla pivot levels (S3/S4 and R3/R4) from 1d timeframe with momentum confirmation (RSI divergence) and volume filter to capture reversals in both bull and bear markets. Uses 1w EMA200 trend filter to avoid counter-trend entries. Designed for low trade frequency (15-25/year) to minimize fee decay while capturing high-probability reversal setups.
"""

name = "4h_Camarilla_Pivot_Reversal_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI (14) for momentum and divergence
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def rma(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First average
            res[period-1] = np.mean(arr[:period])
            # Subsequent values
            for i in range(period, len(arr)):
                res[i] = (arr[i] + (period-1) * res[i-1]) / period
        return res
    
    avg_gain = rma(gain, 14)
    avg_loss = rma(loss, 14)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_s3 = np.zeros_like(close_1d)
    camarilla_s4 = np.zeros_like(close_1d)
    camarilla_r3 = np.zeros_like(close_1d)
    camarilla_r4 = np.zeros_like(close_1d)
    pivot = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        range_ = high_1d[i] - low_1d[i]
        if range_ <= 0:
            camarilla_s3[i] = camarilla_s4[i] = camarilla_r3[i] = camarilla_r4[i] = pivot[i] = close_1d[i]
        else:
            pivot[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
            camarilla_s3[i] = pivot[i] - 1.1 * range_ / 6
            camarilla_s4[i] = pivot[i] - 1.1 * range_ / 2
            camarilla_r3[i] = pivot[i] + 1.1 * range_ / 6
            camarilla_r4[i] = pivot[i] + 1.1 * range_ / 2
    
    # Align Camarilla levels to 4h timeframe (with 1-bar delay for completed daily bar)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_4h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r4_4h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Get 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation (20-period average)
    def sma(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            for i in range(period-1, len(arr)):
                res[i] = np.mean(arr[i-period+1:i+1])
        return res
    
    vol_ma = sma(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # RSI warmup + volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(rsi[i]) or np.isnan(s3_4h[i]) or np.isnan(s4_4h[i]) or \
           np.isnan(r3_4h[i]) or np.isnan(r4_4h[i]) or np.isnan(pivot_4h[i]) or \
           np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirm = volume[i] > 1.3 * vol_ma[i] if vol_ma[i] > 0 else False
        
        # RSI conditions for momentum
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        
        if position == 0:
            # Long setup: price near S3/S4 with bullish momentum
            near_support = (close[i] <= s3_4h[i] * 1.002 and close[i] >= s4_4h[i] * 0.998) or \
                          (close[i] <= s4_4h[i] * 1.002 and close[i] >= s4_4h[i] * 0.998)
            bullish_momentum = rsi[i] > 30 and rsi[i] < rsi[i-1] * 1.05  # RSI rising from oversold
            
            if near_support and volume_confirm and bullish_momentum and close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short setup: price near R3/R4 with bearish momentum
            elif close[i] >= r3_4h[i] * 0.998 and close[i] <= r4_4h[i] * 1.002:
                near_resistance = True
            else:
                near_resistance = False
                
            bearish_momentum = rsi[i] < 70 and rsi[i] > rsi[i-1] * 0.95  # RSI falling from overbought
            
            if near_resistance and volume_confirm and bearish_momentum and close[i] < ema_200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches pivot or RSI shows weakness
            if close[i] >= pivot_4h[i] * 0.998 or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches pivot or RSI shows strength
            if close[i] <= pivot_4h[i] * 1.002 or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals