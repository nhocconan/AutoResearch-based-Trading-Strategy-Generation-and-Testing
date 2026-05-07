#!/usr/bin/env python3
name = "6h_1d_RSI21_SMA100_Rebound"
timeframe = "6h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily SMA(100) for trend filter
    sma_100_1d = pd.Series(df_1d['close']).rolling(window=100, min_periods=100).mean().values
    sma_100_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_100_1d)
    
    # 6h RSI(21) for entry
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/21, min_periods=21, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/21, min_periods=21, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # 6h volume spike filter (2x 12-bar average = 3 days)
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 21, 12)  # Wait for SMA, RSI, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(sma_100_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near SMA(100) support with RSI oversold bounce and volume
            near_sma = abs(close[i] - sma_100_1d_aligned[i]) / sma_100_1d_aligned[i] < 0.02
            rsi_oversold = rsi[i] < 30
            rsi_rising = rsi[i] > rsi[i-1]
            vol_spike = volume[i] > vol_ma_12[i] * 2.0
            
            if near_sma and rsi_oversold and rsi_rising and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # Short: price near SMA(100) resistance with RSI overbought rejection and volume
            elif near_sma and rsi[i] > 70 and rsi[i] < rsi[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI overbought or price moves far from SMA
            if rsi[i] > 70 or abs(close[i] - sma_100_1d_aligned[i]) / sma_100_1d_aligned[i] > 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI oversold or price moves far from SMA
            if rsi[i] < 30 or abs(close[i] - sma_100_1d_aligned[i]) / sma_100_1d_aligned[i] > 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h RSI(21) rebound from daily SMA(100) with volume confirmation
# - Daily SMA(100) acts as major support/resistance in both bull and bear markets
# - In uptrends, price pulls back to SMA(100) and bounces (long opportunity)
# - In downtrends, price rallies to SMA(100) and gets rejected (short opportunity)
# - RSI(21) < 30 with rising momentum identifies oversold bounces
# - RSI(21) > 70 with falling momentum identifies overbought rejections
# - Volume spike (2x average) confirms institutional participation at key levels
# - Works in ranging markets too as price oscillates around SMA(100)
# - Position size 0.25 limits drawdown while capturing meaningful moves
# - Target: 15-30 trades/year to avoid fee drag on 6h timeframe