#!/usr/bin/env python3
"""
1d MACD Signal with 1w Trend Filter and Volume Confirmation
Hypothesis: MACD provides momentum signals on daily timeframe. Filtered by weekly EMA trend to avoid counter-trend trades and volume confirmation to avoid false signals. Works in bull/bear by aligning with higher timeframe trend. Targets 10-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_macd_signal_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = df_1w['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # MACD (12, 26, 9)
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_12 - ema_26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(35, n):  # Start after MACD warmup (26+9)
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(macd_line[i]) or 
            np.isnan(signal_line[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: MACD crosses below signal line OR trend turns bearish
            if (macd_line[i] < signal_line[i] or 
                close[i] < ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: MACD crosses above signal line OR trend turns bullish
            if (macd_line[i] > signal_line[i] or 
                close[i] > ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: MACD bullish crossover, uptrend, volume
            if (macd_line[i] > signal_line[i] and 
                macd_line[i-1] <= signal_line[i-1] and  # crossover confirmation
                close[i] > ema_50_1w_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: MACD bearish crossover, downtrend, volume
            elif (macd_line[i] < signal_line[i] and 
                  macd_line[i-1] >= signal_line[i-1] and  # crossover confirmation
                  close[i] < ema_50_1w_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals