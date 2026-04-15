#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend with 1d volume confirmation and 1w RSI filter
# Designed for low trade frequency (target 15-25/year) with clear trend following logic
# Uses adaptive trend (KAMA) to avoid whipsaws, volume to confirm strength, and RSI to filter extremes
# Works in both bull (trend continuation) and bear (trend continuation) markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for KAMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # KAMA parameters (adaptive moving average)
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close_4h, n=1))
    change = np.concatenate([[np.nan], change])
    volatility = np.abs(np.diff(close_4h, n=1))
    volatility = np.concatenate([[np.nan], volatility])
    
    # Rolling sum for volatility
    vol_sum = pd.Series(volatility).rolling(window=er_period, min_periods=er_period).sum().values
    change_sum = pd.Series(change).rolling(window=er_period, min_periods=er_period).sum().values
    
    # Avoid division by zero
    er = np.divide(change_sum, vol_sum, out=np.zeros_like(change_sum), where=vol_sum!=0)
    
    # Smoothing constant
    sc = np.power(er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1), 2)
    
    # Calculate KAMA
    kama = np.full_like(close_4h, np.nan)
    kama[er_period] = close_4h[er_period]  # Start with first valid value
    for i in range(er_period + 1, len(close_4h)):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # RSI on 1w (14-period)
    delta = np.diff(close_1w)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR for volatility and stoploss (14-period on 4h)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    tr1 = np.maximum(high_4h[1:], low_4h[:-1]) - np.minimum(high_4h[1:], low_4h[:-1])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(atr_aligned[i])):
            continue
        
        # Volatility-adjusted position size (inverse vol)
        vol_factor = np.clip(0.5 * atr_aligned[i] / (close[i] + 1e-10), 0.5, 2.0)
        position_size = base_size / vol_factor
        position_size = np.clip(position_size, 0.15, 0.35)
        
        # Long entry: price above KAMA + RSI not overbought + volume spike
        if (close[i] > kama_aligned[i] and 
            rsi_aligned[i] < 70 and 
            volume[i] > 1.5 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price below KAMA + RSI not oversold + volume spike
        elif (close[i] < kama_aligned[i] and 
              rsi_aligned[i] > 30 and 
              volume[i] > 1.5 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: reverse signal or price crosses KAMA
        elif position == 1 and close[i] < kama_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > kama_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_KAMA_1dVolume_1wRSI_Trend"
timeframe = "4h"
leverage = 1.0