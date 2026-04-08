# 12h Donchian Breakout with 1W Trend Filter
# Hypothesis: Price breaks out of Donchian channels on 12h chart when weekly trend is strong (ADX>25),
# and mean-reverts at channel boundaries when weekly trend is weak (ADX<25).
# Uses volatility-adjusted position sizing to manage risk in both bull and bear markets.
# Target: 20-40 trades/year per symbol.

name = "12h_donchian_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter - call ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 14-period ADX for weekly
    # True Range
    tr1_1w = high_1w[1:] - low_1w[1:]
    tr2_1w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_1w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))])
    
    # Directional Movement
    dm_plus_1w = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                          np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus_1w = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                           np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus_1w = np.concatenate([[0], dm_plus_1w])
    dm_minus_1w = np.concatenate([[0], dm_minus_1w])
    
    # Smoothed values
    tr14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    dm_plus_14_1w = pd.Series(dm_plus_1w).rolling(window=14, min_periods=14).sum().values
    dm_minus_14_1w = pd.Series(dm_minus_1w).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus_1w = 100 * dm_plus_14_1w / tr14_1w
    di_minus_1w = 100 * dm_minus_14_1w / tr14_1w
    
    # DX and ADX
    dx_1w = 100 * np.abs(di_plus_1w - di_minus_1w) / (di_plus_1w + di_minus_1w)
    adx_1w = pd.Series(dx_1w).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(adx_1w[i]):
            signals[i] = 0.0
            continue
        
        # Get aligned weekly ADX for current 12h bar
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)[i]
        
        # Regime detection
        strong_trend_1w = adx_1w_aligned > 25
        
        if position == 1:  # Long position
            # Exit conditions
            if strong_trend_1w:
                # In strong trend: exit on lower Donchian break (stop loss)
                if i >= 20 and low[i] < np.min(low[i-20:i]):
                    position = 0
                    signals[i] = 0.0
            else:
                # In weak trend: exit on upper Donchian (take profit)
                if i >= 20 and high[i] > np.max(high[i-20:i]):
                    position = 0
                    signals[i] = 0.0
                # Or exit on lower Donchian break (stop loss)
                elif i >= 20 and low[i] < np.min(low[i-20:i]):
                    position = 0
                    signals[i] = 0.0
            if position == 1:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit conditions
            if strong_trend_1w:
                # In strong trend: exit on upper Donchian break (stop loss)
                if i >= 20 and high[i] > np.max(high[i-20:i]):
                    position = 0
                    signals[i] = 0.0
            else:
                # In weak trend: exit on lower Donchian (take profit)
                if i >= 20 and low[i] < np.min(low[i-20:i]):
                    position = 0
                    signals[i] = 0.0
                # Or exit on upper Donchian break (stop loss)
                elif i >= 20 and high[i] > np.max(high[i-20:i]):
                    position = 0
                    signals[i] = 0.0
            if position == -1:
                signals[i] = -0.30
        else:  # Flat, look for entry
            if i < 20:
                signals[i] = 0.0
                continue
                
            # Calculate Donchian channels (20-period)
            upper_channel = np.max(high[i-20:i])
            lower_channel = np.min(low[i-20:i])
            
            # Entry logic based on regime
            if strong_trend_1w:
                # Strong trend: breakout entries
                if close[i] > upper_channel and close[i-1] <= upper_channel:
                    position = 1
                    signals[i] = 0.30
                elif close[i] < lower_channel and close[i-1] >= lower_channel:
                    position = -1
                    signals[i] = -0.30
            else:
                # Weak trend: mean reversion at channel boundaries
                if close[i] < lower_channel and close[i-1] >= lower_channel:
                    position = 1
                    signals[i] = 0.30
                elif close[i] > upper_channel and close[i-1] <= upper_channel:
                    position = -1
                    signals[i] = -0.30
    
    return signals