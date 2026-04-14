#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 1d ADX regime filter
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# In ranging markets (ADX < 20): fade extremes (sell Bull Power > 0, buy Bear Power < 0)
# In trending markets (ADX > 25): follow momentum (buy Bull Power > 0, sell Bear Power < 0)
# Uses 6-period EMA for responsiveness on 6H chart
# Target: 50-120 trades over 4 years with regime adaptation for 2021-2026

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (6-period EMA for responsiveness)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate 1d ADX (14-period) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ADX
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(13, n):
        # Get aligned 1d ADX
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(adx_1d_aligned):
            continue
        
        # Regime classification
        ranging = adx_1d_aligned < 20
        trending = adx_1d_aligned > 25
        
        if position == 0:  # No position - look for entries
            if ranging:
                # In range: fade extremes
                if bull_power[i] > 0:  # Overbought - sell
                    position = -1
                    signals[i] = -position_size
                elif bear_power[i] > 0:  # Oversold - buy
                    position = 1
                    signals[i] = position_size
            elif trending:
                # In trend: follow momentum
                if bull_power[i] > 0:  # Bullish momentum - buy
                    position = 1
                    signals[i] = position_size
                elif bear_power[i] > 0:  # Bearish momentum - sell
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit conditions
            # Exit if momentum fades or reverses
            if bull_power[i] <= 0:  # Lost bullish momentum
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit conditions
            # Exit if momentum fades or reverses
            if bear_power[i] <= 0:  # Lost bearish momentum
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1dADX_Regime"
timeframe = "6h"
leverage = 1.0