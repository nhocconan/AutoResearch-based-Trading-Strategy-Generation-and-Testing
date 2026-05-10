#!/usr/bin/env python3
# 1d_Pullback_to_200SMA_with_200MA_Volume_Spike
# Hypothesis: On the daily chart, buy when price pulls back to the 200-day SMA during an uptrend
# (defined as price above 200-day SMA) with volume confirmation (volume > 2x 20-day average).
# Sell when price closes above the 20-day EMA (taking profits on short-term strength).
# Short when price rallies to the 200-day SMA during a downtrend (price below 200-day SMA)
# with volume confirmation, and cover when price closes below the 20-day EMA.
# The 200-day SMA acts as dynamic support/resistance, and volume spikes confirm institutional interest.
# Designed for low trade frequency (10-25/year) to minimize fee dust.

name = "1d_Pullback_to_200SMA_with_200MA_Volume_Spike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 200-day SMA for trend and support/resistance
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # 20-day EMA for entry/exit timing
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 20-day volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter (1w) - use only uptrend/downtrend, no counter-trend
    df_1w = get_htf_data(prices, '1w')
    sma_50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need 200 days for SMA200
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if np.isnan(sma_200[i]) or np.isnan(ema_20[i]) or np.isnan(vol_ma_20[i]) or np.isnan(sma_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: volume > 2x 20-day average
        vol_confirm = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 0:
            # Long: price near 200-day SMA (within 1%), uptrend (price > SMA200), weekly uptrend, volume spike
            if (abs(close[i] - sma_200[i]) / sma_200[i] < 0.01 and 
                close[i] > sma_200[i] and 
                close[i] > sma_50_1w_aligned[i] and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price near 200-day SMA (within 1%), downtrend (price < SMA200), weekly downtrend, volume spike
            elif (abs(close[i] - sma_200[i]) / sma_200[i] < 0.01 and 
                  close[i] < sma_200[i] and 
                  close[i] < sma_50_1w_aligned[i] and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes above 20-day EMA (take profit on short-term strength)
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes below 20-day EMA (take profit on short-term weakness)
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals