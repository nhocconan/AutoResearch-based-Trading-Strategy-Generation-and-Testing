#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation
# and volatility filter. Works in bull/bear by capturing breakouts with institutional
# volume while avoiding choppy markets. Target: 15-25 trades/year.
name = "1d_Weekly_Donchian_Breakout_Volume_VolatilityFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_max_20 = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe
    donchian_high = align_htf_to_ltf(prices, df_weekly, high_max_20)
    donchian_low = align_htf_to_ltf(prices, df_weekly, low_min_20)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid extremely high volatility (>90th percentile)
    # Use ATR(14) normalized by price
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ratio = atr_14 / close
    atr_ratio_90 = pd.Series(atr_ratio).rolling(window=50, min_periods=50).quantile(0.90).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(atr_ratio_90[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        vol_ratio = vol / vol_ma if vol_ma > 0 else 0
        
        # Entry conditions
        long_entry = (price > donchian_high[i] and 
                     vol_ratio > 1.5 and 
                     atr_ratio[i] < atr_ratio_90[i])
        short_entry = (price < donchian_low[i] and 
                      vol_ratio > 1.5 and 
                      atr_ratio[i] < atr_ratio_90[i])
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below weekly Donchian low
            if price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above weekly Donchian high
            if price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals