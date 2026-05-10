#!/usr/bin/env python3
# 6h_Momentum_Divergence_Strategy
# Hypothesis: Combines RSI divergence with volume confirmation and 1-day trend filter.
# In bull markets, we go long on bullish RSI divergence (price makes LL, RSI makes HL) with volume confirmation.
# In bear markets, we go short on bearish RSI divergence (price makes HH, RSI makes LH) with volume confirmation.
# Uses 1-day EMA20 as trend filter to align with higher timeframe momentum.
# Designed for low trade frequency (15-25/year) to minimize fee drag.

name = "6h_Momentum_Divergence_Strategy"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Daily EMA20 for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # RSI calculation (14-period)
    def rsi(arr, period=14):
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(arr, np.nan)
        avg_loss = np.full_like(arr, np.nan)
        
        # First average
        if len(arr) >= period + 1:
            avg_gain[period] = np.mean(gain[1:period+1])
            avg_loss[period] = np.mean(loss[1:period+1])
            
            # Wilder's smoothing
            for i in range(period+1, len(arr)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Volume confirmation (20-period average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 10  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(ema_20_1d_aligned[i]) or np.isnan(rsi_vals[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            # Look back 3 periods for pivot points
            if i >= 3:
                price_ll = low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i-3]
                price_hh = high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i-3]
                rsi_hl = rsi_vals[i] > rsi_vals[i-1] and rsi_vals[i] > rsi_vals[i-2] and rsi_vals[i] > rsi_vals[i-3]
                rsi_lh = rsi_vals[i] < rsi_vals[i-1] and rsi_vals[i] < rsi_vals[i-2] and rsi_vals[i] < rsi_vals[i-3]
                
                # Long: bullish divergence in downtrend (price LL, RSI HL) with volume confirmation and above daily EMA20
                if price_ll and rsi_hl and volume_confirm and close[i] > ema_20_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish divergence in uptrend (price HH, RSI LH) with volume confirmation and below daily EMA20
                elif price_hh and rsi_lh and volume_confirm and close[i] < ema_20_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price makes new high or RSI becomes overbought
            if high[i] > high[i-1] and high[i] > high[i-2] or rsi_vals[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price makes new low or RSI becomes oversold
            if low[i] < low[i-1] and low[i] < low[i-2] or rsi_vals[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals