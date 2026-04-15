#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 12h volume confirmation and 1d trend filter
# Designed for low trade frequency (target 20-30/year) with mean reversion in trending markets
# Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
# Uses Williams %R for entry timing and volume spike for confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for Williams %R calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Williams %R (14-period) on 4h
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low + 1e-10)
    
    # Load 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    
    # Load 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Volume average (20-period on 12h)
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR for volatility and position sizing (14-period on 4h)
    tr1 = np.maximum(high_4h[1:], low_4h[:-1]) - np.minimum(high_4h[1:], low_4h[:-1])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    vol_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_avg_12h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_aligned[i])):
            continue
        
        # Volatility-adjusted position size (inverse vol)
        vol_factor = np.clip(0.5 * atr_aligned[i] / (close[i] + 1e-10), 0.5, 2.0)
        position_size = base_size / vol_factor
        position_size = np.clip(position_size, 0.15, 0.35)
        
        # Long entry: Williams %R oversold (< -80) + uptrend + volume spike
        if (williams_r_aligned[i] < -80 and 
            close[i] > ema50_1d_aligned[i] and 
            volume[i] > 2.0 * vol_avg_12h_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: Williams %R overbought (> -20) + downtrend + volume spike
        elif (williams_r_aligned[i] > -20 and 
              close[i] < ema50_1d_aligned[i] and 
              volume[i] > 2.0 * vol_avg_12h_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: reverse signal or Williams %R returns to neutral range
        elif position == 1 and (williams_r_aligned[i] > -50 or 
                                close[i] < ema50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (williams_r_aligned[i] < -50 or 
                                 close[i] > ema50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsR_12hVolume_1dEMA_MeanReversion"
timeframe = "4h"
leverage = 1.0