#!/usr/bin/env python3
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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(21) for trend filter
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 1d RSI(14) for momentum filter
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 6h Donchian(20) channels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Daily EMA21 uptrend (price above EMA21)
        # 2. Daily RSI between 40-60 (neutral momentum, avoiding extremes)
        # 3. Price breaks above 6h Donchian(20) high with volume confirmation
        if (close[i] > ema_21_1d_aligned[i] and
            40 <= rsi_14_1d_aligned[i] <= 60 and
            close[i] > donchian_high_20[i] and
            volume_ratio[i] > 1.5):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Daily EMA21 downtrend (price below EMA21)
        # 2. Daily RSI between 40-60 (neutral momentum, avoiding extremes)
        # 3. Price breaks below 6h Donchian(20) low with volume confirmation
        elif (close[i] < ema_21_1d_aligned[i] and
              40 <= rsi_14_1d_aligned[i] <= 60 and
              close[i] < donchian_low_20[i] and
              volume_ratio[i] > 1.5):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_EMA21_RSI40-60_Donchian20_VolumeBreakout_v1"
timeframe = "6h"
leverage = 1.0