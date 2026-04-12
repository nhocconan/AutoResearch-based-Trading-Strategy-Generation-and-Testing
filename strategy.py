#!/usr/bin/env python3
# 4h_1d_Triple_Barrier_Breakout_v1
# Hypothesis: Combine 1d volatility bands (ATR-based) with 4h price action and volume confirmation.
# Long when price breaks above 1d ATR(20) upper band with volume > 1.8x 20-period average,
# short when breaks below 1d ATR(20) lower band with volume > 1.8x 20-period average.
# Exit when price returns to 1d close or ATR band midpoint.
# Designed for low trade frequency (<50/year) to minimize fee drift in ranging markets.
# Works in bull via breakouts above volatility expansion, in bear via breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Triple_Barrier_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for volatility bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:  # Need 20 for ATR + 1 for prev close
        return np.zeros(n)
    
    # Calculate ATR(20) on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Daily bands: upper = close + 2.0*ATR, lower = close - 2.0*ATR, mid = close
    upper_band = close_1d + 2.0 * atr
    lower_band = close_1d - 2.0 * atr
    mid_band = close_1d
    
    # Align daily bands to 4h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    mid_band_aligned = align_htf_to_ltf(prices, df_1d, mid_band)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # default to 1.0 if no MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(mid_band_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume filter
        long_breakout = close[i] > upper_band_aligned[i] and vol_ratio[i] > 1.8
        short_breakout = close[i] < lower_band_aligned[i] and vol_ratio[i] > 1.8
        
        # Exit conditions: return to daily midpoint
        long_exit = close[i] < mid_band_aligned[i]
        short_exit = close[i] > mid_band_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals