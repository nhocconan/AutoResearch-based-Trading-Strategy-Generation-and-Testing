#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Stochastic_Trend_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1-day EMA20 for trend
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # 6h Stochastic %K (14,3,3)
    low_min = pd.Series(low).rolling(window=14, min_periods=14).min()
    high_max = pd.Series(high).rolling(window=14, min_periods=14).max()
    stoch_k = 100 * (close - low_min) / (high_max - low_min)
    stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan).fillna(50)
    # Smooth %K with 3-period SMA
    stoch_k_smooth = stoch_k.rolling(window=3, min_periods=3).mean()
    # %D is 3-period SMA of smoothed %K
    stoch_d = stoch_k_smooth.rolling(window=3, min_periods=3).mean()
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(stoch_k_smooth.iloc[i]) or 
            np.isnan(stoch_d.iloc[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        stoch_k_val = stoch_k_smooth.iloc[i]
        stoch_d_val = stoch_d.iloc[i]
        
        if position == 0:
            # Long: Stochastic bullish crossover in oversold (<20) + uptrend + volume
            long_cond = (stoch_k_val > stoch_d_val and 
                        stoch_k_val < 20 and 
                        stoch_k_smooth.iloc[i-1] <= stoch_d.iloc[i-1] and  # crossover just happened
                        close[i] > ema_20_1d_aligned[i] and
                        volume_filter[i])
            
            # Short: Stochastic bearish crossover in overbought (>80) + downtrend + volume
            short_cond = (stoch_k_val < stoch_d_val and 
                         stoch_k_val > 80 and 
                         stoch_k_smooth.iloc[i-1] >= stoch_d.iloc[i-1] and  # crossover just happened
                         close[i] < ema_20_1d_aligned[i] and
                         volume_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Stochastic bearish crossover OR price below EMA
            if (stoch_k_val < stoch_d_val and 
                stoch_k_smooth.iloc[i-1] >= stoch_d.iloc[i-1]) or \
               close[i] < ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Stochastic bullish crossover OR price above EMA
            if (stoch_k_val > stoch_d_val and 
                stoch_k_smooth.iloc[i-1] <= stoch_d.iloc[i-1]) or \
               close[i] > ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals