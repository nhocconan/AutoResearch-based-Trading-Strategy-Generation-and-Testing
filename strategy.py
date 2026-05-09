# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Uses Camarilla R1/S1 from 1d as price channel, 1d EMA50 for trend filter,
# volume > 1.5x 20-period average for confirmation. Positions held until
# price closes back inside R1-S1 range or trend flips.
# Position size: 0.28
# Designed to work in both bull and bear markets via trend filter + range-bound reversals.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    # Get 1d data for Camarilla pivot levels and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels: R1, S1
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 (using previous day's range)
    # R1 = Close + 1.1 * (High - Low) / 12
    # S1 = Close - 1.1 * (High - Low) / 12
    # Using previous day's values to avoid look-ahead
    range_1d = high_1d - low_1d
    r1_raw = close_1d + 1.1 * range_1d / 12
    s1_raw = close_1d - 1.1 * range_1d / 12
    
    # Align 1d Camarilla levels to 4h timeframe (waits for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_raw)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_raw)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 (resistance) with bullish trend and volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.28
                position = 1
            # Enter short: price breaks below S1 (support) with bearish trend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.28
                position = -1
        
        elif position == 1:
            # Exit long: price closes back inside R1-S1 range OR trend turns bearish
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:
            # Exit short: price closes back inside R1-S1 range OR trend turns bullish
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals