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
    
    # Load daily data for HL2-based PPO - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 35:
        return np.zeros(n)
    
    # Calculate daily HL2
    hl2_daily = (df_daily['high'].values + df_daily['low'].values) / 2
    
    # Calculate PPO on daily HL2: (EMA12 - EMA26) / EMA26 * 100
    ema12 = pd.Series(hl2_daily).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(hl2_daily).ewm(span=26, adjust=False, min_periods=26).mean().values
    ppo_daily = np.where(ema26 != 0, (ema12 - ema26) / ema26 * 100, 0)
    
    # Calculate signal line (9-period EMA of PPO)
    ppo_signal = pd.Series(ppo_daily).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate PPO histogram
    ppo_hist = ppo_daily - ppo_signal
    
    # Calculate 60-period SMA of PPO histogram for trend filter
    ppo_hist_sma = pd.Series(ppo_hist).rolling(window=60, min_periods=60).mean().values
    
    # Align PPO histogram and its SMA to 6h timeframe
    ppo_hist_aligned = align_htf_to_ltf(prices, df_daily, ppo_hist)
    ppo_hist_sma_aligned = align_htf_to_ltf(prices, df_daily, ppo_hist_sma)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ppo_hist_aligned[i]) or np.isnan(ppo_hist_sma_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: PPO histogram crosses above its SMA with volume confirmation
            if (ppo_hist_aligned[i] > ppo_hist_sma_aligned[i] and 
                ppo_hist_aligned[i-1] <= ppo_hist_sma_aligned[i-1] and
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: PPO histogram crosses below its SMA with volume confirmation
            elif (ppo_hist_aligned[i] < ppo_hist_sma_aligned[i] and 
                  ppo_hist_aligned[i-1] >= ppo_hist_sma_aligned[i-1] and
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: PPO histogram crosses back below/above its SMA
            if position == 1:
                if ppo_hist_aligned[i] < ppo_hist_sma_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if ppo_hist_aligned[i] > ppo_hist_sma_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_PPO_Histogram_SMA_Crossover_Volume_Filter"
timeframe = "6h"
leverage = 1.0