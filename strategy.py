# 4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Strict_v1
# Hypothesis: Camarilla pivot levels (R1/S1) from daily act as institutional support/resistance.
# Breakouts with volume confirmation and trend filter (EMA34) yield high-probability moves.
# Works in bull/bear: breakouts capture momentum; volume filter avoids fakeouts.
# Target: 20-40 trades/year per symbol (<160 total over 4 years) to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    R1 = close_1d + (range_ * 1.1 / 12)
    S1 = close_1d - (range_ * 1.1 / 12)
    R2 = close_1d + (range_ * 1.1 / 6)
    S2 = close_1d - (range_ * 1.1 / 6)
    
    # Calculate 34-period EMA on daily for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume moving average for volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 4h timeframe (primary)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_4h = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_4h[i]
        
        # Trend filter: price above/below EMA34
        uptrend = close[i] > ema_34_4h[i]
        downtrend = close[i] < ema_34_4h[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and uptrend
            if close[i] > R1_4h[i] and volume_filter and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and downtrend
            elif close[i] < S1_4h[i] and volume_filter and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below S1 OR trend reverses
            if (close[i] < S1_4h[i]) or (not uptrend):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 OR trend reverses
            if (close[i] > R1_4h[i]) or (not downtrend):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Strict_v1"
timeframe = "4h"
leverage = 1.0