#!/usr/bin/env python3
"""
1-day Donchian(15) breakout with 1-week RSI filter and volume confirmation
Hypothesis: Breakouts of Donchian(15) channels on daily timeframe in the direction of 
weekly RSI trend, confirmed by volume > 2x 20-period average, capture momentum with 
few whipsaws. Designed for ~15-25 trades/year to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_rsi_volume_v1"
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
    
    # 1-week data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1-week RSI(14) for trend filter
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1w = (100 - (100 / (1 + rs))).values
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_1w_aligned[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI turns bearish (<50) OR price breaks below Donchian(10) low
            donchian_low = np.min(low[max(0, i-10):i+1])
            if (rsi_14_1w_aligned[i] < 50 or 
                close[i] <= donchian_low):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI turns bullish (>50) OR price breaks above Donchian(10) high
            donchian_high = np.max(high[max(0, i-10):i+1])
            if (rsi_14_1w_aligned[i] > 50 or 
                close[i] >= donchian_high):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Donchian(15) channels - slightly tighter for fewer trades
            donchian_high = np.max(high[max(0, i-15):i])
            donchian_low = np.min(low[max(0, i-15):i])
            
            # Long: price breaks above Donchian(15) high + volume spike + RSI > 50
            if (close[i] > donchian_high and
                rsi_14_1w_aligned[i] > 50 and
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian(15) low + volume spike + RSI < 50
            elif (close[i] < donchian_low and
                  rsi_14_1w_aligned[i] < 50 and
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals