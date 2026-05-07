#!/usr/bin/env python3
"""
4H_Camarilla_R1_S1_Breakout_1DTrend_VolumeS_Rev3
Hypothesis: Add momentum filter (Stochastic RSI) to reduce whipsaws and overtrading while maintaining 
the proven edge of Camarilla breakout with volume confirmation. Use StochRSI < 20 for long (oversold) 
and > 80 for short (overbought) to enter only at momentum extremes. This should reduce false breakouts 
and improve win rate in both bull and bear markets by avoiding entries during weak momentum.
Target: 50-150 trades/year on 4H timeframe with disciplined entries.
"""
name = "4H_Camarilla_R1_S1_Breakout_1DTrend_VolumeS_Rev3"
timeframe = "4h"
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
    
    # Get 1D data for Camarilla levels and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = close + (high - low) * 1.1 / 12, S1 = close - (high - low) * 1.1 / 12
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: current 4h volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    # Stochastic RSI (14,14,3,3) to identify momentum extremes
    rsi_period = 14
    stoch_period = 14
    k_period = 3
    d_period = 3
    
    # Calculate RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate Stochastic RSI
    rsi_min = pd.Series(rsi_values).rolling(window=stoch_period, min_periods=stoch_period).min()
    rsi_max = pd.Series(rsi_values).rolling(window=stoch_period, min_periods=stoch_period).max()
    stoch_rsi = (rsi_values - rsi_min) / (rsi_max - rsi_min) * 100
    # Handle division by zero when rsi_max == rsi_min
    stoch_rsi = np.where(rsi_max == rsi_min, 50, stoch_rsi)
    
    # Calculate %K and %D
    k = pd.Series(stoch_rsi).rolling(window=k_period, min_periods=k_period).mean()
    d = pd.Series(k).rolling(window=d_period, min_periods=d_period).mean()
    stoch_k = k.values
    stoch_d = d.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14, 14, 3, 3)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg[i]) or 
            np.isnan(stoch_k[i]) or 
            np.isnan(stoch_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above daily EMA34, 4h close above daily R1, volume confirmation, and StochRSI K < 20 (oversold)
            if (close[i] > ema_34_1d_aligned[i] and 
                close[i] > r1_aligned[i] and 
                volume_filter[i] and 
                stoch_k[i] < 20):
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA34, 4h close below daily S1, volume confirmation, and StochRSI K > 80 (overbought)
            elif (close[i] < ema_34_1d_aligned[i] and 
                  close[i] < s1_aligned[i] and 
                  volume_filter[i] and 
                  stoch_k[i] > 80):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below daily EMA34
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above daily EMA34
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals