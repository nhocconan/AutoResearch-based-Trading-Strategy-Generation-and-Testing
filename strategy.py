#!/usr/bin/env python3
# 1d_Weekly_Range_Breakout_Volume
# Hypothesis: Combines weekly range breakouts with volume confirmation and RSI filter.
# Uses weekly high/low for entry, volume spike for confirmation, and RSI to avoid overextended moves.
# Designed to work in both trending and ranging markets by capturing breakouts from weekly ranges.
# Target: 10-25 trades/year per symbol with disciplined risk to avoid fee drag.

name = "1d_Weekly_Range_Breakout_Volume"
timeframe = "1d"
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
    
    # Get weekly data for range calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly high and low
    weekly_high = high_1w
    weekly_low = low_1w
    
    # Align weekly levels to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Calculate RSI(14) on daily closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average of first 14 periods
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: daily volume / 20-day average volume
    vol_ma = np.zeros_like(volume)
    vol_ma[:19] = np.nan
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.divide(volume, vol_ma, out=np.full_like(volume, np.nan), where=vol_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need RSI and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or \
           np.isnan(rsi[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above weekly high AND volume confirmation AND RSI not overbought
            if close[i] > weekly_high_aligned[i] and volume_ratio[i] > 1.8 and rsi[i] < 70:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below weekly low AND volume confirmation AND RSI not oversold
            elif close[i] < weekly_low_aligned[i] and volume_ratio[i] > 1.8 and rsi[i] > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below weekly low OR RSI overbought
            if close[i] < weekly_low_aligned[i] or rsi[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price breaks above weekly high OR RSI oversold
            if close[i] > weekly_high_aligned[i] or rsi[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals