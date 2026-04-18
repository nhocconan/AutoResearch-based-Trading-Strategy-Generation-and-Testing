#!/usr/bin/env python3
"""
4h_1d_MACD_Signal_Crossover_Volume
Hypothesis: Use MACD line crossing above/below signal line as momentum signal on 4h, confirmed by volume and 1d trend (price above/below 200 EMA). MACD captures momentum shifts effectively, while 1d EMA200 filters for higher-timeframe trend alignment. Volume > 1.3x 20-period average confirms conviction. Designed for ~25-35 trades/year by requiring MACD crossover, volume confirmation, and 1d trend filter. Works in bull markets via long signals when MACD turns up above signal with price > EMA200, and in bear markets via short signals when MACD turns down below signal with price < EMA200.
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
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 200 EMA on 1d close
    ema200_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema200_1d[i] = (close_1d[i] * 2 / (200 + 1)) + (ema200_1d[i-1] * (1 - 2 / (200 + 1)))
    
    # Align EMA200 to 4h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Get 4h data for MACD calculation
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate MACD: 12 EMA - 26 EMA, signal line = 9 EMA of MACD
    if len(close_4h) >= 26:
        # EMA 12
        ema12 = np.full_like(close_4h, np.nan)
        ema12[11] = np.mean(close_4h[:12])
        for i in range(12, len(close_4h)):
            ema12[i] = (close_4h[i] * 2 / (12 + 1)) + (ema12[i-1] * (1 - 2 / (12 + 1)))
        
        # EMA 26
        ema26 = np.full_like(close_4h, np.nan)
        ema26[25] = np.mean(close_4h[:26])
        for i in range(26, len(close_4h)):
            ema26[i] = (close_4h[i] * 2 / (26 + 1)) + (ema26[i-1] * (1 - 2 / (26 + 1)))
        
        # MACD line
        macd_line = ema12 - ema26
        
        # Signal line: 9 EMA of MACD
        signal_line = np.full_like(macd_line, np.nan)
        valid_macd = ~np.isnan(macd_line)
        if np.sum(valid_macd) >= 9:
            # Find first valid index for seeding
            first_valid = np.where(valid_macd)[0][0]
            signal_line[first_valid + 8] = np.mean(macd_line[first_valid:first_valid + 9])
            for i in range(first_valid + 9, len(macd_line)):
                if not np.isnan(macd_line[i]):
                    signal_line[i] = (macd_line[i] * 2 / (9 + 1)) + (signal_line[i-1] * (1 - 2 / (9 + 1)))
    else:
        macd_line = np.full_like(close_4h, np.nan)
        signal_line = np.full_like(close_4h, np.nan)
    
    # Align MACD and signal line to 4h timeframe
    macd_line_aligned = align_htf_to_ltf(prices, df_4h, macd_line)
    signal_line_aligned = align_htf_to_ltf(prices, df_4h, signal_line)
    
    # Volume confirmation: current volume > 1.3 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 20)  # need MACD and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(macd_line_aligned[i]) or np.isnan(signal_line_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: MACD crosses above signal line, with volume, and 1d uptrend (price > EMA200)
            if (macd_line_aligned[i] > signal_line_aligned[i] and 
                macd_line_aligned[i-1] <= signal_line_aligned[i-1] and
                vol_confirm[i] and 
                close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: MACD crosses below signal line, with volume, and 1d downtrend (price < EMA200)
            elif (macd_line_aligned[i] < signal_line_aligned[i] and 
                  macd_line_aligned[i-1] >= signal_line_aligned[i-1] and
                  vol_confirm[i] and 
                  close[i] < ema200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: MACD crosses below signal line (momentum shift) or price crosses below EMA200 (trend change)
            if (macd_line_aligned[i] < signal_line_aligned[i] and 
                macd_line_aligned[i-1] >= signal_line_aligned[i-1]) or \
               close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: MACD crosses above signal line (momentum shift) or price crosses above EMA200 (trend change)
            if (macd_line_aligned[i] > signal_line_aligned[i] and 
                macd_line_aligned[i-1] <= signal_line_aligned[i-1]) or \
               close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_MACD_Signal_Crossover_Volume"
timeframe = "4h"
leverage = 1.0