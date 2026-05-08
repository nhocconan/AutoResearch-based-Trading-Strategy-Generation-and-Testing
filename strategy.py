#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Stochastic_Pullback_1dTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h Stochastic %K (14,3) - overbought/oversold
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    stoch_k = ((close - lowest_low_14) / (highest_high_14 - lowest_low_14 + 1e-10)) * 100
    
    # 4h SMA of %K for signal smoothing
    stoch_k_sma = pd.Series(stoch_k).rolling(window=3, min_periods=3).mean().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(stoch_k_sma[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Stochastic oversold (<20) + uptrend (price > 1d EMA34) + volume spike
            long_cond = (stoch_k_sma[i] < 20) and \
                        (close[i] > ema_34_1d_aligned[i]) and \
                        volume_spike[i]
            # Short: Stochastic overbought (>80) + downtrend (price < 1d EMA34) + volume spike
            short_cond = (stoch_k_sma[i] > 80) and \
                         (close[i] < ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Stochastic overbought (>80) or trend reversal
            if stoch_k_sma[i] > 80 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Stochastic oversold (<20) or trend reversal
            if stoch_k_sma[i] < 20 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals