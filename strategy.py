#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend + RSI mean reversion + 12h volume spike filter
# KAMA adapts to market noise - follows trends but avoids whipsaws in ranging markets
# RSI(14) < 30/70 provides mean reversion entries in the direction of KAMA trend
# 12h volume spike (>2x average) confirms institutional participation
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# Low turnover expected: ~20-30 trades/year per symbol

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for volume spike filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate KAMA (adaptive moving average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    kama_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    
    # Handle arrays properly
    er = np.zeros_like(change)
    er[10:] = change[10:] / volatility[10:]  # Avoid div by zero
    er = np.concatenate([np.full(10, np.nan), er])
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.full_like(close, np.nan)
    kama[kama_len] = close[kama_len]  # Seed
    
    for i in range(kama_len + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    rsi_len = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_len, min_periods=rsi_len).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_len, min_periods=rsi_len).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(rsi_len, np.nan), rsi])
    
    # Calculate 12h volume average and spike filter
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = vol_12h > 2.0 * vol_ma_12h
    
    # Align 12h volume spike to 4h timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h.astype(float))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, kama_len + 1, rsi_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Volume confirmation: 12h volume spike
        volume_confirmed = vol_spike_aligned[i] > 0.5  # Boolean as float
        
        if position == 0:
            # Enter long: price > KAMA (uptrend) + RSI < 30 (oversold) + volume spike
            if (price_above_kama and 
                rsi[i] < 30 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price < KAMA (downtrend) + RSI > 70 (overbought) + volume spike
            elif (price_below_kama and 
                  rsi[i] > 70 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 70 (overbought) or price crosses below KAMA
            if (rsi[i] > 70 or 
                close[i] < kama[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 30 (oversold) or price crosses above KAMA
            if (rsi[i] < 30 or 
                close[i] > kama[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_KAMA_RSI_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0