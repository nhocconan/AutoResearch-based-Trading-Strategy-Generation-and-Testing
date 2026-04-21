# [EXPERIMENT #72008] 12h_Donchian_Breakout_WeeklyTrend_Filter
# Hypothesis: Donchian(20) breakout on 12h with weekly EMA20 trend filter and volume confirmation.
# Works in bull (breakouts with trend) and bear (avoids counter-trend breakouts via weekly filter).
# Target: 12-37 trades/year (50-150 over 4 years) to avoid fee drag.
# Uses 1w trend filter (EMA20) to avoid false breakouts in choppy markets.
# Position size: 0.25 (discrete to minimize fee churn).
# Risk: Exit on reverse breakout or volatility expansion (weekly ATR ratio > 2.5).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Donchian channel and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Weekly EMA20 for trend filter ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === Daily Donchian Channel (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper and lower bands
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe (use previous day's values)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # === Daily ATR for volatility filter ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR ratio: current ATR / 50-period average ATR
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        ema_20_1w_val = ema_20_1w_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian with volume, above weekly EMA20, and moderate volatility
            if (price_close > upper and 
                price_close > ema_20_1w_val and 
                vol_ratio_val > 1.5 and 
                atr_ratio_val > 0.8 and atr_ratio_val < 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian with volume, below weekly EMA20, and moderate volatility
            elif (price_close < lower and 
                  price_close < ema_20_1w_val and 
                  vol_ratio_val > 1.5 and 
                  atr_ratio_val > 0.8 and atr_ratio_val < 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse breakout or volatility expansion/contraction
            if position == 1 and (price_close < lower or atr_ratio_val > 2.5 or atr_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > upper or atr_ratio_val > 2.5 or atr_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_WeeklyTrend_Filter"
timeframe = "12h"
leverage = 1.0