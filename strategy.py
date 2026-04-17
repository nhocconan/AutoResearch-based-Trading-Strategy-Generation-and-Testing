#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and range calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range (high - low)
    daily_range = high_1d - low_1d
    
    # Calculate 10-day ATR of daily ranges (volatility filter)
    tr1 = daily_range
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr_daily = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr_daily).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 5-day average of daily ATR (to avoid division by zero and smooth)
    atr_daily_ma5 = pd.Series(atr_daily).rolling(window=5, min_periods=5).mean().values
    
    # Align daily ATR and its 5-day MA to 4h timeframe
    atr_daily_aligned = align_htf_to_ltf(prices, df_1d, atr_daily)
    atr_daily_ma5_aligned = align_htf_to_ltf(prices, df_1d, atr_daily_ma5)
    
    # Calculate 4-hour ATR for entry/exit logic
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr1_4h[0] = np.nan
    tr2_4h[0] = np.nan
    tr3_4h[0] = np.nan
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period SMA of 4h ATR for volatility regime filter
    atr_ma20 = pd.Series(atr_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_daily_aligned[i]) or 
            np.isnan(atr_daily_ma5_aligned[i]) or
            np.isnan(atr_4h[i]) or 
            np.isnan(atr_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: daily ATR > 5-day MA of daily ATR (avoid low volatility periods)
        vol_regime_filter = atr_daily_aligned[i] > atr_daily_ma5_aligned[i]
        # 4h volatility filter: current ATR > 20-period MA of ATR (avoid choppy 4h periods)
        vol_4h_filter = atr_4h[i] > atr_ma20[i]
        
        if position == 0:
            # Long: bullish momentum in high volatility regime
            if vol_regime_filter and vol_4h_filter and close[i] > close[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: bearish momentum in high volatility regime
            elif vol_regime_filter and vol_4h_filter and close[i] < close[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: momentum fails or volatility drops
            if close[i] <= close[i-1] or not vol_4h_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: momentum fails or volatility drops
            if close[i] >= close[i-1] or not vol_4h_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VolRegime_Momentum_Simple"
timeframe = "4h"
leverage = 1.0