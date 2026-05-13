# 165135
#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Reversal_With_Volume_Filter_v2
Hypothesis: Weekly pivot reversals work in both bull and bear markets. 
Price rejecting weekly R1/S1 with volume confirmation and contrarian positioning 
(extreme RSI) captures reversals. Weekly pivot provides structure, RSI(2) 
identifies exhaustion, volume filter ensures conviction. Designed for 15-30 trades/year.
"""

name = "6h_Weekly_Pivot_Reversal_With_Volume_Filter_v2"
timeframe = "6h"
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
    
    # Get weekly data for pivot calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points from previous week
    # Typical Price = (H + L + C) / 3
    typical_price = (df_weekly['high'] + df_weekly['low'] + df_weekly['close']) / 3
    # Weekly range
    weekly_range = df_weekly['high'] - df_weekly['low']
    
    # Weekly pivot levels
    weekly_pivot = typical_price
    weekly_r1 = (2 * weekly_pivot) - df_weekly['low']
    weekly_s1 = (2 * weekly_pivot) - df_weekly['high']
    weekly_r2 = weekly_pivot + weekly_range
    weekly_s2 = weekly_pivot - weekly_range
    
    # Align to 6h - use previous week's levels (available at 6h open)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1.values)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1.values)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2.values)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2.values)
    
    # RSI(2) for exhaustion signals
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price rejects S1 (bounces off support) with exhaustion RSI and volume
            if (low[i] <= weekly_s1_aligned[i] * 1.002 and  # Allow small penetration
                close[i] > weekly_s1_aligned[i] and         # Close back above S1
                rsi_values[i] < 30 and                      # Oversold
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price rejects R1 (fails at resistance) with exhaustion RSI and volume
            elif (high[i] >= weekly_r1_aligned[i] * 0.998 and  # Allow small penetration
                  close[i] < weekly_r1_aligned[i] and         # Close back below R1
                  rsi_values[i] > 70 and                      # Overbought
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R1 (take profit) or RSI normalizes
            if (close[i] >= weekly_r1_aligned[i] or 
                rsi_values[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S1 (take profit) or RSI normalizes
            if (close[i] <= weekly_s1_aligned[i] or 
                rsi_values[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals