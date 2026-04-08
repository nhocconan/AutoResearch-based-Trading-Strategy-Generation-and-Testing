#!/usr/bin/env python3
# 4h_camilla_pivot_volume_reversal_v1
# Hypothesis: Uses 1d Camarilla pivot levels (L3, H3) for mean reversion entries
# with volume confirmation and 4h momentum filter. Works in both bull/bear
# as pivots act as support/resistance in ranging markets and trend continuations
# in trending markets. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camilla_pivot_volume_reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h RSI(14) for momentum filter
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    for i in range(rsi_period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    # H2 = close + 0.75*(high-low), L2 = close - 0.75*(high-low)
    # H1 = close + 0.5*(high-low), L1 = close - 0.5*(high-low)
    # Pivot = (high + low + close)/3
    camarilla_H3 = np.zeros(len(close_1d))
    camarilla_L3 = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        high_low = high_1d[i-1] - low_1d[i-1]
        camarilla_H3[i] = close_1d[i-1] + 1.125 * high_low
        camarilla_L3[i] = close_1d[i-1] - 1.125 * high_low
    
    # Align Camarilla levels to 4h timeframe
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(rsi_period, vol_ma_period, 1) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price reaches H3 or RSI > 70 (overbought)
            if close[i] >= camarilla_H3_aligned[i] or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price reaches L3 or RSI < 30 (oversold)
            if close[i] <= camarilla_L3_aligned[i] or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price touches L3 with volume surge and RSI < 40 (not overbought)
            if (close[i] <= camarilla_L3_aligned[i] and 
                vol_surge[i] and 
                rsi[i] < 40):
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches H3 with volume surge and RSI > 60 (not oversold)
            elif (close[i] >= camarilla_H3_aligned[i] and 
                  vol_surge[i] and 
                  rsi[i] > 60):
                position = -1
                signals[i] = -0.25
    
    return signals