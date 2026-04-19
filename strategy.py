#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Elder Ray Index with 1d volume confirmation and volatility regime filter.
# Long when Bull Power > 0 AND Bear Power < 0 AND volume > 1.5x 20-period 1d average volume AND ATR(14) < ATR(50) (low volatility)
# Short when Bear Power < 0 AND Bull Power > 0 AND volume > 1.5x 20-period 1d average volume AND ATR(14) < ATR(50)
# Exit when Bull Power and Bear Power cross (Bull Power < 0 for longs, Bear Power > 0 for shorts)
# Uses Elder Ray to measure bull/bear power relative to EMA, volume for conviction, volatility regime to avoid chop.
# Target: 20-30 trades/year per symbol.
name = "4h_ElderRay_Volume_VolatilityFilter"
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
    
    # Get 1d data for volume and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 13-period EMA for Elder Ray (using 1d close)
    close_1d = df_1d['close'].values
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema13
    bear_power = df_1d['low'].values - ema13
    
    # Align Elder Ray components to 4h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Get 1d ATR for volatility regime filter (14 and 50 periods)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    # Get 1d average volume for confirmation (20-period)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 50)  # Ensure EMA13 and ATR50 are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(atr14_aligned[i]) or np.isnan(atr50_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        bp = bull_power_aligned[i]
        br = bear_power_aligned[i]
        atr14_val = atr14_aligned[i]
        atr50_val = atr50_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        
        # Volatility regime: only trade when ATR14 < ATR50 (low volatility)
        vol_regime = atr14_val < atr50_val
        
        if position == 0:
            # Long entry: Bull Power > 0 AND Bear Power < 0 AND volume spike AND low volatility
            if bp > 0 and br < 0 and vol > 1.5 * vol_ma and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0 AND Bull Power > 0 AND volume spike AND low volatility
            elif br < 0 and bp > 0 and vol > 1.5 * vol_ma and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power crosses below zero
            if bp < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power crosses above zero
            if br > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals