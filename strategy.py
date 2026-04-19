#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R1/S1 Breakout with 1d Trend Filter and Volume Confirmation
# Camarilla pivot levels provide strong intraday support/resistance levels
# 1d EMA34 trend filter ensures trading in direction of higher timeframe trend
# Volume confirmation filters weak breakouts and confirms institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
name = "12h_Camarilla_R1S1_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Typical price = (High + Low + Close) / 3
    typical_price = (high + low + close) / 3
    range_ = high - low
    
    # Camarilla levels
    R4 = close + (range_ * 1.1 / 2)
    R3 = close + (range_ * 1.1 / 4)
    R2 = close + (range_ * 1.1 / 6)
    R1 = close + (range_ * 1.1 / 12)
    S1 = close - (range_ * 1.1 / 12)
    S2 = close - (range_ * 1.1 / 6)
    S3 = close - (range_ * 1.1 / 4)
    S4 = close - (range_ * 1.1 / 2)
    
    # Shift levels to avoid look-ahead (use previous bar's levels)
    R1 = np.roll(R1, 1)
    S1 = np.roll(S1, 1)
    R2 = np.roll(R2, 1)
    S2 = np.roll(S2, 1)
    R1[0] = np.nan
    S1[0] = np.nan
    R2[0] = np.nan
    S2[0] = np.nan
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isclose(close[i], 0) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with trend up and volume confirmation
            if (close[i] > R1[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with trend down and volume confirmation
            elif (close[i] < S1[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 or trend turns down
            if (close[i] < S1[i]) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 or trend turns up
            if (close[i] > R1[i]) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals