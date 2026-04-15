#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + 1d RSI Mean Reversion + Volume Spike
# Long when price breaks above Donchian(20) high, RSI(14) < 40, and volume > 1.5x 20-bar median.
# Short when price breaks below Donchian(20) low, RSI(14) > 60, and volume > 1.5x 20-bar median.
# Exit when price crosses the 20-period SMA or RSI reverts to neutral (40-60).
# Uses daily RSI for mean reversion bias, works in both bull (breakouts) and bear (mean reversion) markets.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day RSI(14) for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # 4h Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h SMA(20) for exit
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(sma_20[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Donchian breakout up, RSI oversold (<40), volume spike
        if (close[i] > high_20[i] and 
            rsi_1d_aligned[i] < 40 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Donchian breakout down, RSI overbought (>60), volume spike
        elif (close[i] < low_20[i] and 
              rsi_1d_aligned[i] > 60 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses 20-SMA or RSI returns to neutral (40-60)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < sma_20[i] or rsi_1d_aligned[i] >= 40)) or
               (signals[i-1] == -0.25 and (close[i] > sma_20[i] or rsi_1d_aligned[i] <= 60)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_RSI1d_Volume"
timeframe = "4h"
leverage = 1.0