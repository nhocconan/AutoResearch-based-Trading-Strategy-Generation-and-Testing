#!/usr/bin/env python3
"""
4h_1d_Pivot_R1S1_Breakout_Volume_Regime_v2
- Uses daily Camarilla pivot levels (R1/S1) for breakout entries
- Volume confirmation: current 4h volume > 1.5x average 4h volume
- Regime filter: 1d ADX > 25 to trade only in strong trends
- Position size: 0.25 for clear risk control
- Target: 20-40 trades/year to minimize fee drag
- Works in bull/bear by only trading strong trends (ADX filter)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Pivot_R1S1_Breakout_Volume_Regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    R1 = typical_price + (range_hl * 1.1 / 12)
    S1 = typical_price - (range_hl * 1.1 / 12)
    
    # Align pivot levels to 4h timeframe (use previous day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1.values)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1.values)
    
    # Calculate ADX(14) on daily data for trend strength
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = df_1d['high'].diff()
    minus_dm = df_1d['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume filter: current 4h volume > 1.5x average 4h volume (20-period)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter
        volume_filter = vol_ma_4h[i] > 0 and volume[i] > 1.5 * vol_ma_4h[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and strong trend
            if (close[i] > R1_aligned[i] and 
                volume_filter and 
                adx_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S1 with volume and strong trend
            elif (close[i] < S1_aligned[i] and 
                  volume_filter and 
                  adx_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: price breaks below S1 or trend weakens
            if (close[i] < S1_aligned[i] or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price breaks above R1 or trend weakens
            if (close[i] > R1_aligned[i] or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals