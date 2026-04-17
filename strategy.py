#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Channels with 1-day RSI Filter - Uses 1D RSI to filter 4H breakouts
# Works in bull markets by taking breakouts in strong momentum; works in bear by avoiding overbought/oversold false signals
# RSI filter prevents entries during exhaustion; breakout captures momentum continuations
# Target: 15-30 trades/year to minimize fee drag while capturing significant moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h Donchian channel (15-period) ===
    high_15 = pd.Series(close).rolling(window=15, min_periods=15).max().values
    low_15 = pd.Series(close).rolling(window=15, min_periods=15).min().values
    
    # === 1d RSI(14) for momentum filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # RSI calculation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === Volume confirmation ===
    vol_ma_10_4h = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(high_15[i]) or np.isnan(low_15[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_10_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_15[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_15[i-1]  # Break below previous period's low
        
        # Volume filter: current 4h volume > 1.3x 10-period average
        vol_filter = volume[i] > vol_ma_10_4h[i] * 1.3
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Donchian breakout up + RSI not overbought + volume filter
            if breakout_up and rsi_1d_aligned[i] < 70 and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Donchian breakout down + RSI not oversold + volume filter
            elif breakout_down and rsi_1d_aligned[i] > 30 and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long when RSI becomes overbought or price returns to middle
            mid_channel = (high_15[i] + low_15[i]) / 2
            if rsi_1d_aligned[i] > 75 or close[i] < mid_channel:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when RSI becomes oversold or price returns to middle
            mid_channel = (high_15[i] + low_15[i]) / 2
            if rsi_1d_aligned[i] < 25 or close[i] > mid_channel:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian15_1dRSI_Vol1.3x"
timeframe = "4h"
leverage = 1.0