# 1. Hypothesis
# The strategy uses a 4-hour primary timeframe with 1-day and 1-week higher timeframes.
# It combines three key elements: 1) Weekly EMA trend filter to capture multi-week bias,
# 2) Daily ATR-based volatility regime filter to avoid choppy markets, and 
# 3) Volume confirmation to ensure institutional participation.
# Entries occur when price crosses the weekly EMA with sufficient volume and 
# moderate volatility. Exits occur on reverse crossovers or volatility extremes.
# This approach aims to capture sustained trends while avoiding false signals 
# in low-volume or choppy conditions, working in both bull and bear markets 
# by following the higher timeframe trend.

# 2. Implementation
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily ATR for volatility filter and stop-loss
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volatility ratio: current ATR / 50-period average ATR
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: volume / 20-day average volume
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        weekly_ema = ema_34_1w_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price above weekly EMA, moderate volatility, volume spike
            if (price_close > weekly_ema and 
                vol_ratio_val > 1.5 and 
                atr_ratio_val > 0.8 and atr_ratio_val < 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price below weekly EMA, moderate volatility, volume spike
            elif (price_close < weekly_ema and 
                  vol_ratio_val > 1.5 and 
                  atr_ratio_val > 0.8 and atr_ratio_val < 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse crossover or volatility extremes
            if position == 1 and (price_close < weekly_ema or atr_ratio_val > 2.5 or atr_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > weekly_ema or atr_ratio_val > 2.5 or atr_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyEMA34_Volume_ATR_Filter"
timeframe = "1d"
leverage = 1.0