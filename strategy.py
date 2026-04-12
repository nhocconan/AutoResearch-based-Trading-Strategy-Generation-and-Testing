#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray (trend filter)
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13_1d  # High - EMA13
    bear_power = low - ema13_1d   # Low - EMA13
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 22-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(21, n):
        atr[i] = np.nanmean(tr[i-21:i+1])
    
    # Calculate 22-period ATR EMA for volatility regime
    atr_ema = np.full(n, np.nan)
    atr_series = pd.Series(atr)
    atr_ema_values = atr_series.ewm(span=22, adjust=False, min_periods=22).mean().values
    atr_ema[:] = atr_ema_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ema[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 1.2x 22-period ATR EMA (elevated volatility)
        vol_filter = atr[i] > atr_ema[i] * 1.2
        
        # Elder Ray signals: bull/bear power crossing zero with volatility expansion
        bull_cross_up = bull_power_aligned[i] > 0 and bull_power_aligned[i-1] <= 0
        bear_cross_down = bear_power_aligned[i] < 0 and bear_power_aligned[i-1] >= 0
        
        # Entry conditions: Elder Ray cross with volatility expansion
        long_entry = bull_cross_up and vol_filter
        short_entry = bear_cross_down and vol_filter
        
        # Exit conditions: reverse Elder Ray cross or volatility contraction
        long_exit = bear_power_aligned[i] > 0 and bear_power_aligned[i-1] <= 0
        short_exit = bull_power_aligned[i] < 0 and bull_power_aligned[i-1] >= 0
        vol_exit = atr[i] < atr_ema[i] * 0.8  # volatility contraction
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (long_exit or vol_exit):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (short_exit or vol_exit):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_power_vol_filter_v1"
timeframe = "6h"
leverage = 1.0