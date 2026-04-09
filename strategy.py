# 1d_1w_volatility_breakout_v3
# Hypothesis: Weekly volatility expansion signals strong directional moves. Breakouts from weekly Donchian channels with volume confirmation capture trending moves in both bull and bear markets. Daily timeframe ensures lower trade frequency (target: 10-30 trades/year) to minimize fee drag. Uses volatility regime filter to avoid false breakouts in low-volatility environments.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_volatility_breakout_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Donchian channels and volatility
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    high_max = pd.Series(df_w['high']).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(df_w['low']).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly ATR(14) for volatility regime filter
    tr1 = pd.Series(df_w['high']).shift(1) - pd.Series(df_w['low']).shift(1)
    tr2 = abs(pd.Series(df_w['high']).shift(1) - pd.Series(df_w['close']))
    tr3 = abs(pd.Series(df_w['low']).shift(1) - pd.Series(df_w['close']))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_w = tr.rolling(window=14, min_periods=14).mean().values
    
    # Align weekly values to daily timeframe
    high_max_aligned = align_htf_to_ltf(prices, df_w, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_w, low_min)
    atr_w_aligned = align_htf_to_ltf(prices, df_w, atr_w)
    
    # Volume confirmation: 20-day average volume
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(high_max_aligned[i]) or 
            np.isnan(low_min_aligned[i]) or 
            np.isnan(atr_w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: only trade when weekly ATR is above its 50-period average
        atr_ma_50 = np.full(n, np.nan)
        atr_sum = 0
        for j in range(i+1):
            if not np.isnan(atr_w_aligned[j]):
                atr_sum += atr_w_aligned[j]
                if j >= 50:
                    # Find the first valid ATR value to subtract
                    k = j - 50
                    while k >= 0 and np.isnan(atr_w_aligned[k]):
                        k -= 1
                    if k >= 0:
                        atr_sum -= atr_w_aligned[k]
        if i >= 49:
            valid_count = 0
            atr_sum_valid = 0
            for j in range(i-49, i+1):
                if not np.isnan(atr_w_aligned[j]):
                    atr_sum_valid += atr_w_aligned[j]
                    valid_count += 1
            if valid_count >= 50:
                atr_ma_50[i] = atr_sum_valid / 50
        
        if position == 1:  # Long position
            # Exit: price closes below weekly midpoint
            midpoint = (high_max_aligned[i] + low_min_aligned[i]) / 2.0
            if close[i] < midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly midpoint
            midpoint = (high_max_aligned[i] + low_min_aligned[i]) / 2.0
            if close[i] > midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volatility filter: only trade when current ATR > 50-day average ATR
            vol_filter = not np.isnan(atr_ma_50[i]) and atr_w_aligned[i] > atr_ma_50[i]
            
            # Enter long: price closes above weekly Donchian high with volume and volatility confirmation
            if (vol_filter and
                close[i] > high_max_aligned[i] and 
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below weekly Donchian low with volume and volatility confirmation
            elif (vol_filter and
                  close[i] < low_min_aligned[i] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals