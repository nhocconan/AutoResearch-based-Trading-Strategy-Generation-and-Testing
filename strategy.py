# 1d_Leverage_ETF_Pairs_Rotation - 1d strategy for BTC/ETH/SOL pairs rotation
# Hypothesis: Rotates between BTC, ETH, and SOL based on relative strength and momentum
# Uses 1d RSI, 20-day SMA, and volume to identify strongest performer
# Weekly trend filter to ensure alignment with higher timeframe momentum
# Designed to work in both bull and bear markets by always holding the strongest asset
# Target: 20-40 trades/year to minimize fee drag while capturing momentum shifts

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
    
    # Get weekly data for trend filter (SMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate SMA(50) on weekly close
    sma_50_1w = np.full(len(df_1w), np.nan)
    for i in range(49, len(close_1w)):
        sma_50_1w[i] = np.mean(close_1w[i-49:i+1])
    
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate daily indicators
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 20-day SMA
    sma_20 = np.full(n, np.nan)
    for i in range(19, n):
        sma_20[i] = np.mean(close[i-19:i+1])
    
    # Volume average (20-day)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    volume_ratio = np.divide(volume, vol_ma_20, out=np.full_like(volume, np.nan), where=vol_ma_20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    # Warmup: need enough data for indicators
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(sma_50_1w_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(sma_20[i]) or
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from weekly SMA50
        if i > 0 and not np.isnan(sma_50_1w_aligned[i-1]):
            trend_up = close[i] > sma_50_1w_aligned[i]
        else:
            trend_up = False
        
        if position == 0:
            # Long entry: price above weekly SMA50 + RSI > 50 + volume above average
            if (trend_up and 
                rsi[i] > 50 and 
                volume_ratio[i] > 1.2):
                signals[i] = 0.30
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price below weekly SMA50 OR RSI < 40
            if (close[i] <= sma_50_1w_aligned[i] or 
                rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
    
    return signals

name = "1d_Leverage_ETF_Pairs_Rotation_v1"
timeframe = "1d"
leverage = 1.0