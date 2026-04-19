#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with volume confirmation and ATR filter
# - Use 1d Camarilla pivot levels (R1, R2, S1, S2) as key support/resistance
# - Long when price breaks above R2 with volume > 1.5x 20-period average
# - Short when price breaks below S2 with volume > 1.5x 20-period average
# - ATR(14) filter: only trade when ATR > 0.5 * 20-period ATR average (avoid chop)
# - Exit when price returns to pivot point (PP) or opposite S1/R1 level
# - Designed to capture institutional breakouts with follow-through
# - Target: 15-30 trades/year to minimize fee drag while capturing strong moves

name = "6h_Camarilla_R2_S2_Breakout_Volume_ATRFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1/12
    # S1 = C - (H - L) * 1.1/12
    # R2 = C + (H - L) * 1.1/6
    # S2 = C - (H - L) * 1.1/6
    # R3 = C + (H - L) * 1.1/4
    # S3 = C - (H - L) * 1.1/4
    # R4 = C + (H - L) * 1.1/2
    # S4 = C - (H - L) * 1.1/2
    
    # We need previous day's data to calculate today's pivot levels
    # So we shift the high, low, close by 1 to get previous day's values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels using previous day's data
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    r2 = close_1d + (high_1d - low_1d) * 1.1 / 6.0
    s2 = close_1d - (high_1d - low_1d) * 1.1 / 6.0
    
    # Align pivot levels to 6t timeframe (wait for previous day's close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter to avoid choppy markets
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or np.isnan(atr_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = vol_ma_20[i] > 0 and volume[i] > 1.5 * vol_ma_20[i]
        
        # ATR filter: only trade when ATR > 0.5 * 20-period ATR average (avoid chop)
        atr_filter = atr[i] > 0.5 * atr_ma_20[i]
        
        if position == 0:
            # Long breakout: price breaks above R2 with volume and ATR filter
            if close[i] > r2_aligned[i] and volume_filter and atr_filter:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S2 with volume and ATR filter
            elif close[i] < s2_aligned[i] and volume_filter and atr_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to pivot point or S1
            if close[i] <= pp_aligned[i] or close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to pivot point or R1
            if close[i] >= pp_aligned[i] or close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals