#!/usr/bin/env python3
"""
1d_Stochastic_RSI_Trend_Signal
Hypothesis: Use Stochastic RSI on daily timeframe to identify overbought/oversold conditions with trend confirmation from 1-week EMA. Enter long when StochRSI crosses above 20 in an uptrend (price > weekly EMA), short when StochRSI crosses below 80 in a downtrend (price < weekly EMA). Uses volume confirmation to avoid false signals. Designed for low frequency (target 10-20 trades/year) to minimize fee drag while capturing major trend reversals in both bull and bear markets.
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
    
    # Get 1d data for Stochastic RSI
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Stochastic RSI (14, 14, 3, 3)
    rsi_period = 14
    stoch_period = 14
    k_period = 3
    d_period = 3
    
    # RSI calculation
    def calculate_rsi(close_prices, period):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
        
        for i in range(rsi_period+1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close_1d, rsi_period)
    
    # Stochastic of RSI
    stoch_rsi = np.full_like(close_1d, np.nan)
    for i in range(stoch_period-1, len(rsi)):
        min_rsi = np.min(rsi[i-stoch_period+1:i+1])
        max_rsi = np.max(rsi[i-stoch_period+1:i+1])
        if max_rsi - min_rsi != 0:
            stoch_rsi[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi) * 100
        else:
            stoch_rsi[i] = 50.0
    
    # %K and %D
    k = np.full_like(close_1d, np.nan)
    d = np.full_like(close_1d, np.nan)
    for i in range(k_period-1, len(stoch_rsi)):
        k[i] = np.mean(stoch_rsi[i-k_period+1:i+1])
    for i in range(d_period-1, len(k)):
        d[i] = np.mean(k[i-d_period+1:i+1])
    
    # Get 1w data for trend filter (EMA)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA (20-period)
    ema_1w = np.full_like(close_1w, np.nan)
    multiplier = 2 / (20 + 1)
    ema_1w[19] = np.mean(close_1w[:20])
    for i in range(20, len(close_1w)):
        ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Align indicators to 1d timeframe
    stoch_k_aligned = align_htf_to_ltf(prices, df_1d, k)
    stoch_d_aligned = align_htf_to_ltf(prices, df_1d, d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # need Stochastic and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(stoch_k_aligned[i]) or np.isnan(stoch_d_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: StochRSI crosses above 20, price above weekly EMA, with volume
            if (stoch_k_aligned[i] > 20 and stoch_k_aligned[i-1] <= 20 and
                close[i] > ema_1w_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: StochRSI crosses below 80, price below weekly EMA, with volume
            elif (stoch_k_aligned[i] < 80 and stoch_k_aligned[i-1] >= 80 and
                  close[i] < ema_1w_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: StochRSI crosses above 80 (overbought) or trend change
            if (stoch_k_aligned[i] > 80 and stoch_k_aligned[i-1] <= 80) or \
               close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: StochRSI crosses below 20 (oversold) or trend change
            if (stoch_k_aligned[i] < 20 and stoch_k_aligned[i-1] >= 20) or \
               close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Stochastic_RSI_Trend_Signal"
timeframe = "1d"
leverage = 1.0