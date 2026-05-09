#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d ATR-based breakout and volume confirmation
# Uses 1d ATR(14) to define volatility bands for breakout detection
# Volume filter ensures breakouts occur with above-average participation
# Designed to work in both bull (breakouts continue) and bear (false breakdowns reverse) markets
# Target: 50-150 total trades over 4 years to minimize fee drag

name = "12h_ATR_Breakout_Volume_Confirmation"
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
    
    # Get 1d data for ATR calculation and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility measurement
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing (equivalent to RMA)
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])  # Seed with simple average
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # 1d volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    atr_1d_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_avg_1d_12h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate breakout levels using prior 1d ATR
    # Upper band: prior 1d close + 1.5 * ATR
    # Lower band: prior 1d close - 1.5 * ATR
    upper_band = np.roll(close_1d, 1) + 1.5 * atr_1d
    lower_band = np.roll(close_1d, 1) - 1.5 * atr_1d
    
    # Align breakout levels to 12h
    upper_band_12h = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_12h = align_htf_to_ltf(prices, df_1d, lower_band)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need enough data for ATR and volume calculations
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1d_12h[i]) or np.isnan(vol_avg_1d_12h[i]) or 
            np.isnan(upper_band_12h[i]) or np.isnan(lower_band_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > vol_avg_1d_12h[i] * 1.5
        
        if position == 0:
            # Long: break above upper band with volume confirmation
            if close[i] > upper_band_12h[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume confirmation
            elif close[i] < lower_band_12h[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below prior 1d close (mean reversion)
            if close[i] < np.roll(close_1d, 1)[-len(close)+i] if i < len(close) else close[i] < close_1d[-1]:
                # Simplified: exit when price returns to prior 1d close level
                prior_close = np.roll(close_1d, 1)
                prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
                if not np.isnan(prior_close_aligned[i]) and close[i] < prior_close_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above prior 1d close (mean reversion)
            prior_close = np.roll(close_1d, 1)
            prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
            if not np.isnan(prior_close_aligned[i]) and close[i] > prior_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals