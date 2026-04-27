#!/usr/bin/env python3
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
    
    # Get 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close
    close_1d = df_1d['close'].values
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will fix in loop
    
    # Recalculate volatility properly
    volatility = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        volatility[i] = volatility[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    
    er = np.zeros(len(close_1d))
    er[:er_period] = 0
    for i in range(er_period, len(close_1d)):
        if volatility[i] > 0:
            er[i] = np.abs(close_1d[i] - close_1d[i-er_period]) / volatility[i]
        else:
            er[i] = 0
    
    # Calculate SC and KAMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.full(len(close_1d), np.nan)
    kama[er_period] = close_1d[er_period]
    for i in range(er_period + 1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe (already on 1d, just need to align to 1d index)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d close
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    if len(close_1d) >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period-1] = np.mean(loss[1:rsi_period+1])
        for i in range(rsi_period, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rsi = np.full(len(close_1d), 50.0)
    for i in range(rsi_period, len(close_1d)):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Align RSI to 1d
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation on 1d
    vol_ma_period = 20
    vol_ma = np.full(len(close_1d), np.nan)
    for i in range(vol_ma_period, len(close_1d)):
        vol_ma[i] = np.mean(close_1d[i-vol_ma_period:i])  # using close as proxy for volume, will fix
    
    # Actually calculate volume MA
    vol_1d = df_1d['volume'].values
    vol_ma = np.full(len(vol_1d), np.nan)
    for i in range(vol_ma_period, len(vol_1d)):
        vol_ma[i] = np.mean(vol_1d[i-vol_ma_period:i])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup
    start_idx = max(er_period, rsi_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_aligned[i] if vol_ma_aligned[i] > 0 else 0
        
        # Trend: price above/below KAMA
        above_kama = price > kama_aligned[i]
        below_kama = price < kama_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Volume confirmation
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: price above KAMA, RSI oversold, high volume
            if above_kama and rsi_oversold and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: price below KAMA, RSI overbought, high volume
            elif below_kama and rsi_overbought and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below KAMA or RSI overbought
            if below_kama or rsi_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price above KAMA or RSI oversold
            if above_kama or rsi_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_EMA200_RSI_Momentum_VolFilter"
timeframe = "1d"
leverage = 1.0