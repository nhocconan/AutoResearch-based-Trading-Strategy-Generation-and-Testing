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
    
    # Get daily data for pivot points and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (classic)
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    daily_r2 = daily_pivot + (high_1d - low_1d)  # R2 = Pivot + (High - Low)
    daily_s2 = daily_pivot - (high_1d - low_1d)  # S2 = Pivot - (High - Low)
    
    # Align daily pivot levels to 6h timeframe
    daily_pivot_6h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r2_6h = align_htf_to_ltf(prices, df_1d, daily_r2)
    daily_s2_6h = align_htf_to_ltf(prices, df_1d, daily_s2)
    
    # Calculate daily ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need daily ATR14, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(daily_pivot_6h[i]) or 
            np.isnan(daily_r2_6h[i]) or 
            np.isnan(daily_s2_6h[i]) or 
            np.isnan(atr_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Volatility filter: ATR > 0.5 * 20-period average ATR (avoid low volatility chop)
        atr_ma20 = pd.Series(atr_6h).rolling(window=20, min_periods=20).mean()
        atr_ma20_val = atr_ma20.iloc[i] if not np.isnan(atr_ma20.iloc[i]) else 0
        volatility_filter = atr_6h[i] > (0.5 * atr_ma20_val)
        
        # Price relative to daily pivot levels
        price_above_r2 = close[i] > daily_r2_6h[i]
        price_below_s2 = close[i] < daily_s2_6h[i]
        
        if position == 0:
            # Long: Price breaks above daily R2 with volume and volatility
            if (price_above_r2 and volume_filter and volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S2 with volume and volatility
            elif (price_below_s2 and volume_filter and volatility_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily pivot
            if close[i] < daily_pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily pivot
            if close[i] > daily_pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DailyPivot_R2S2_Breakout_Volume_Volatility"
timeframe = "6h"
leverage = 1.0