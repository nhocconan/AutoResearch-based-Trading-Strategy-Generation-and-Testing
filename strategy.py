#!/usr/bin/env python3
"""
6h_1W_1D_Supertrend_Reverse
Hypothesis: Use 1-week Supertrend (ATR 10, multiplier 3) as primary trend filter, combined with 6-hour RSI reversal signals and volume confirmation. The weekly Supertrend captures the longer-term trend direction, reducing whipsaws during 2022 bear market. RSI reversals (RSI < 30 for long, > 70 for short) capture short-term exhaustion within the weekly trend. Volume > 1.5x 20-period average confirms momentum. This combination works in bull markets by taking pullbacks to RSI < 30 in uptrends, and in bear markets by taking bounces to RSI > 70 in downtrends. Targets 15-25 trades/year by requiring weekly trend alignment, RSI extremes, and volume confirmation.
"""

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
    
    # Get 1-week data for Supertrend (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Supertrend on weekly data
    # Parameters: ATR period=10, multiplier=3
    atr_period = 10
    multiplier = 3
    
    # Calculate True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR using Wilder's smoothing
    atr = np.full_like(close_1w, np.nan)
    if len(close_1w) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[1:atr_period])
        for i in range(atr_period, len(close_1w)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate basic upper and lower bands
    hl_avg = (high_1w + low_1w) / 2
    upper_basic = hl_avg + multiplier * atr
    lower_basic = hl_avg - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.full_like(close_1w, np.nan)
    dir_1w = np.full_like(close_1w, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(atr_period, len(close_1w)):
        # Upper band
        if i == atr_period:
            upper_band = upper_basic[i]
            lower_band = lower_basic[i]
        else:
            upper_band = upper_basic[i] if (upper_basic[i] < upper_band or close_1w[i-1] > upper_band) else upper_band
            lower_band = lower_basic[i] if (lower_basic[i] > lower_band or close_1w[i-1] < lower_band) else lower_band
        
        # Trend direction
        if close_1w[i] > upper_band:
            dir_1w[i] = 1
        elif close_1w[i] < lower_band:
            dir_1w[i] = -1
        else:
            dir_1w[i] = dir_1w[i-1]
            if dir_1w[i] == 1 and lower_band < lower_band_prev:
                lower_band = lower_basic[i]
            if dir_1w[i] == -1 and upper_band > upper_band_prev:
                upper_band = upper_basic[i]
        
        # Store values for next iteration
        upper_band_prev = upper_band
        lower_band_prev = lower_band
        
        # Supertrend value
        supertrend[i] = lower_band if dir_1w[i] == 1 else upper_band
    
    # Align Supertrend and direction to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    dir_1w_aligned = align_htf_to_ltf(prices, df_1w, dir_1w)
    
    # Get 1-day data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI on daily data (14-period)
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[1:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[1:rsi_period])
        for i in range(rsi_period, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # need volume MA and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend_aligned[i]) or np.isnan(dir_1w_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: weekly uptrend, RSI < 30 (oversold), volume confirmation
            if (dir_1w_aligned[i] == 1 and rsi_1d_aligned[i] < 30 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend, RSI > 70 (overbought), volume confirmation
            elif (dir_1w_aligned[i] == -1 and rsi_1d_aligned[i] > 70 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: weekly trend turns down or RSI > 70 (overbought)
            if (dir_1w_aligned[i] == -1 or rsi_1d_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend turns up or RSI < 30 (oversold)
            if (dir_1w_aligned[i] == 1 or rsi_1d_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1W_1D_Supertrend_Reverse"
timeframe = "6h"
leverage = 1.0