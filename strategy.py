#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams %R with 1-day ATR-based volatility filter and volume confirmation.
Long when Williams %R crosses above -20 (oversold recovery) + ATR(14) > 1.5 * ATR(50) + volume > 1.5x average.
Short when Williams %R crosses below -80 (overbought breakdown) + ATR(14) > 1.5 * ATR(50) + volume > 1.5x average.
Exit when Williams %R crosses -50 (mean reversion) or volatility drops (ATR(14) < ATR(50)).
Designed for low trade frequency (~15-30/year) to minimize fee drift in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for volatility filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR values to 12h timeframe
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    # Calculate Williams %R (14-period) on 12h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(wr[i]) or np.isnan(atr14_aligned[i]) or np.isnan(atr50_aligned[i]) or 
            np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volatility_expanding = atr14_aligned[i] > 1.5 * atr50_aligned[i]
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: Williams %R crosses above -20 + volatility expanding + volume confirmation
            if (wr[i] > -20 and wr[i-1] <= -20 and 
                volatility_expanding and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 + volatility expanding + volume confirmation
            elif (wr[i] < -80 and wr[i-1] >= -80 and 
                  volatility_expanding and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses below -50 or volatility contracts
                if wr[i] < -50 or not volatility_expanding:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses above -50 or volatility contracts
                if wr[i] > -50 or not volatility_expanding:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_VolatilityFilter_VolumeConfirm"
timeframe = "12h"
leverage = 1.0