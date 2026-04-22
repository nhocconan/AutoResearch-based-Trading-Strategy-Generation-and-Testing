#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1-day RSI and volume confirmation for mean reversion.
# Williams %R(14) > -20 indicates overbought, < -80 indicates oversold.
# RSI(14) on 1-day confirms momentum divergence: RSI < 30 for long, > 70 for short.
# Volume spike filters for conviction. Designed to work in both bull and bear markets
# by fading extremes during pullbacks in trends or mean reversion in ranges.
# Target: 20-30 trades/year per symbol (80-120 total) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R on 6h (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # RSI on 1-day (14-period)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Volume spike filter (24-period on 6h)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 1.5 * vol_ma24
    
    # Align indicators to 6-hour timeframe
    williams_r_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), williams_r)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + RSI < 30 + volume spike
            if (williams_r_aligned[i] < -80 and 
                rsi_1d_aligned[i] < 30 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + RSI > 70 + volume spike
            elif (williams_r_aligned[i] > -20 and 
                  rsi_1d_aligned[i] > 70 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50) or RSI reverts
            if position == 1:
                if (williams_r_aligned[i] > -50 or rsi_1d_aligned[i] > 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (williams_r_aligned[i] < -50 or rsi_1d_aligned[i] < 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_RSI1D_Volume_Spike"
timeframe = "6h"
leverage = 1.0