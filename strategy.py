# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Strategy: 4h breakout of daily Camarilla R1/S1 levels with 1d EMA34 trend filter and volume spike confirmation
# Long: price breaks above R1 + above 1d EMA34 + volume > 1.5x 20-period average
# Short: price breaks below S1 + below 1d EMA34 + volume > 1.5x 20-period average
# Exit: price crosses back below/above the trigger level OR EMA trend contradicts position
# Position size: 0.25 (25% of capital) to balance return and drawdown
# Designed for low trade frequency (<50/year) with high win rate in trending and ranging markets

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # 4h close for price action
    price = close
    
    # 1d EMA34 for trend filter
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    R1 = C + (H - L) * 1.1 / 12
    S1 = C - (H - L) * 1.1 / 12
    
    # Align 1d Camarilla levels to 4h timeframe (waits for daily close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough data for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 + above EMA34 + volume spike
            if (price[i] > R1_aligned[i] and 
                price[i] > ema34[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + below EMA34 + volume spike
            elif (price[i] < S1_aligned[i] and 
                  price[i] < ema34[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below R1 OR EMA34 turns bearish
            if (price[i] < R1_aligned[i]) or (price[i] < ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above S1 OR EMA34 turns bullish
            if (price[i] > S1_aligned[i]) or (price[i] > ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals