# 6h_1d_Camarilla_R1S1_Breakout_Volume_Control
# Hypothesis: Camarilla pivot levels from 1d provide strong intraday support/resistance.
# Breakouts above R1 or below S1 with volume confirmation indicate institutional participation.
# Works in both bull/bear markets by trading breakouts in direction of 1d trend (EMA50).
# Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag.
# Size: 0.25 (conservative to manage drawdown).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR(14) for volume filter (avoid low-vol false breakouts)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume moving average (20-period) for volume spike detection
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Camarilla levels from previous 1d
    # R4 = close + 1.5*(high-low), R3 = close + 1.0*(high-low), etc.
    # We only need R1, S1, R3, S3 for this strategy
    range_1d = high_1d - low_1d
    r1 = close_1d + 1.0833 * range_1d  # R1 = C + 1.0833*(H-L)
    s1 = close_1d - 1.0833 * range_1d  # S1 = C - 1.0833*(H-L)
    r3 = close_1d + 1.2500 * range_1d  # R3 = C + 1.2500*(H-L)
    s3 = close_1d - 1.2500 * range_1d  # S3 = C - 1.2500*(H-L)
    
    # Align Camarilla levels to 6h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period MA (avoid low-vol breakouts)
        vol_spike = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume, above 1d EMA50 (uptrend filter)
            if (close[i] > r1_aligned[i] and vol_spike and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, below 1d EMA50 (downtrend filter)
            elif (close[i] < s1_aligned[i] and vol_spike and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 (failed breakout) or drops below EMA50
            if close[i] < s1_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (failed breakdown) or rises above EMA50
            if close[i] > r1_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Camarilla_R1S1_Breakout_Volume_Control"
timeframe = "6h"
leverage = 1.0