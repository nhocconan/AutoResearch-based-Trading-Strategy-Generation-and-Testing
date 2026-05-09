# 1/10/2025
# Hypothesis: 4h Williams %R reversal with 1d ATR filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. In ranging markets, it
# provides mean-reversion signals. During trends, extreme readings can signal
# continuation when combined with ATR-based volatility filter and volume spike.
# Long when %R < -80 (oversold), ATR expanding, and volume > 1.5x average
# Short when %R > -20 (overbought), ATR contracting, and volume > 1.5x average
# Exit when %R crosses -50 (centerline) or reverses direction
# Designed to work in both trending and ranging markets with controlled frequency
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "4h_WilliamsR_Reversal_ATR_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate True Range and ATR(14)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_1d['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(50).values  # neutral when range=0
    
    # Align 1d indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: oversold (%R < -80), ATR expanding (current > previous), volume spike
            if (williams_r_aligned[i] < -80 and 
                atr_1d_aligned[i] > atr_1d_aligned[i-1] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: overbought (%R > -20), ATR contracting (current < previous), volume spike
            elif (williams_r_aligned[i] > -20 and 
                  atr_1d_aligned[i] < atr_1d_aligned[i-1] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: %R crosses above -50 (centerline) or becomes overbought
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: %R crosses below -50 (centerline) or becomes oversold
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals