#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Stochastic_Trend_Confirmation_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Stochastic %K on 1d data (14,3,3)
    period_k = 14
    period_d = 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    def calculate_stochastic(high_arr, low_arr, close_arr, k_period, d_period):
        n_days = len(close_arr)
        lowest_low = np.full(n_days, np.nan)
        highest_high = np.full(n_days, np.nan)
        percent_k = np.full(n_days, np.nan)
        percent_d = np.full(n_days, np.nan)
        
        for i in range(k_period - 1, n_days):
            lowest_low[i] = np.min(low_arr[i - k_period + 1:i + 1])
            highest_high[i] = np.max(high_arr[i - k_period + 1:i + 1])
            if highest_high[i] != lowest_low[i]:
                percent_k[i] = ((close_arr[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])) * 100
            else:
                percent_k[i] = 50.0
        
        # Calculate %D (SMA of %K)
        k_series = pd.Series(percent_k)
        d_values = k_series.rolling(window=d_period, min_periods=d_period).mean().values
        
        return percent_k, d_values
    
    stoch_k, stoch_d = calculate_stochastic(high_1d, low_1d, close_1d, period_k, period_d)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 6h timeframe
    stoch_k_aligned = align_htf_to_ltf(prices, df_1d, stoch_k)
    stoch_d_aligned = align_htf_to_ltf(prices, df_1d, stoch_d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike on 6h (volume > 1.5 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(stoch_k_aligned[i]) or np.isnan(stoch_d_aligned[i]) or np.isnan(ema_34_aligned[i]):
            signals[i] = 0.0
            continue
            
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when: Stochastic oversold (<20) + %K crosses above %D + above EMA34 + volume spike
            if (stoch_k_aligned[i] < 20 and 
                stoch_k_aligned[i] > stoch_d_aligned[i] and
                close[i] > ema_34_aligned[i] and
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short when: Stochastic overbought (>80) + %K crosses below %D + below EMA34 + volume spike
            elif (stoch_k_aligned[i] > 80 and 
                  stoch_k_aligned[i] < stoch_d_aligned[i] and
                  close[i] < ema_34_aligned[i] and
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when Stochastic overbought or trend changes
            if (stoch_k_aligned[i] > 80 or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when Stochastic oversold or trend changes
            if (stoch_k_aligned[i] < 20 or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals