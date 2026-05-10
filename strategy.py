#!/usr/bin/env python3
# 4h_Top_Bottom_Reversal_With_Volume
# Hypothesis: Price tends to reverse at key daily support/resistance levels (prior day's high/low) 
# when accompanied by volume exhaustion and momentum divergence. Uses daily RSI for 
# overbought/oversold conditions and volume spike for confirmation. Designed for 
# low trade frequency (<30/year) to work in both bull and bear markets by fading 
# overextended moves.

name = "4h_Top_Bottom_Reversal_With_Volume"
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
    
    # Get daily data for key levels and RSI
    df_1d = get_htf_data(prices, '1d')
    # Prior day's high and low (key support/resistance)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    # Daily RSI for overbought/oversold conditions
    rsi_period = 14
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align daily data to 4h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Volume confirmation (20-period average on 4h = ~3.3 days)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or \
           np.isnan(rsi_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8x average (signifies exhaustion)
        volume_confirm = volume[i] > 1.8 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Short: price at/prior day's high with RSI > 70 (overbought) and volume exhaustion
            if close[i] >= 0.99 * prev_high_aligned[i] and rsi_aligned[i] > 70 and volume_confirm:
                signals[i] = -0.25
                position = -1
            # Long: price at/prior day's low with RSI < 30 (oversold) and volume exhaustion
            elif close[i] <= 1.01 * prev_low_aligned[i] and rsi_aligned[i] < 30 and volume_confirm:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Long exit: price moves back above prior day's low or RSI exceeds 50
            if close[i] > prev_low_aligned[i] or rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price moves back below prior day's high or RSI falls below 50
            if close[i] < prev_high_aligned[i] or rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals