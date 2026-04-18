#!/usr/bin/env python3
"""
Hypothesis: 1h MACD histogram crossover with 4h trend filter and volume confirmation.
MACD histogram crossing above/below zero captures momentum shifts. The 4h EMA50 trend filter ensures we trade only in the direction of the higher timeframe trend, reducing whipsaws. Volume confirmation (>1.5x 20-period average) ensures institutional participation. Designed for 15-30 trades/year to minimize fee drag. Works in bull markets (buy when MACD crosses above zero in uptrend) and bear markets (sell when MACD crosses below zero in downtrend).
"""

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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h data
    ema_50_4h = np.full(len(df_4h), np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = (close_4h[i] * 2 / (50 + 1)) + (ema_50_4h[i-1] * (49 / (50 + 1)))
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate MACD (12,26,9) on 1h close
    ema_12 = np.full(n, np.nan)
    ema_26 = np.full(n, np.nan)
    if n >= 26:
        ema_12[11] = np.mean(close[:12])
        ema_26[25] = np.mean(close[:26])
        for i in range(12, n):
            ema_12[i] = (close[i] * 2 / (12 + 1)) + (ema_12[i-1] * (11 / (12 + 1)))
        for i in range(26, n):
            ema_26[i] = (close[i] * 2 / (26 + 1)) + (ema_26[i-1] * (25 / (26 + 1)))
    
    macd_line = ema_12 - ema_26
    
    # Calculate signal line (9-period EMA of MACD)
    signal_line = np.full(n, np.nan)
    valid_macd = ~np.isnan(macd_line)
    if np.sum(valid_macd) >= 9:
        # Find first valid index
        first_valid = np.where(valid_macd)[0][0]
        signal_line[first_valid + 8] = np.mean(macd_line[first_valid:first_valid + 9])
        for i in range(first_valid + 9, n):
            if not np.isnan(macd_line[i]):
                signal_line[i] = (macd_line[i] * 2 / (9 + 1)) + (signal_line[i-1] * (8 / (9 + 1)))
    
    macd_hist = macd_line - signal_line
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 20)  # need MACD and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(macd_hist[i]) or 
            np.isnan(signal_line[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Only trade in the direction of 4h trend with volume confirmation
            if uptrend and vol_confirmed:
                # Long when MACD histogram crosses above zero
                if macd_hist[i] > 0 and macd_hist[i-1] <= 0:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirmed:
                # Short when MACD histogram crosses below zero
                if macd_hist[i] < 0 and macd_hist[i-1] >= 0:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: MACD histogram crosses below zero or trend change
            if macd_hist[i] < 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: MACD histogram crosses above zero or trend change
            if macd_hist[i] > 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1h_MACD_Hist_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0