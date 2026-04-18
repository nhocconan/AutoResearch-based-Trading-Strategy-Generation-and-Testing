#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align daily ATR14 to 12h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 12-period EMA on daily close
    ema_12_1d = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_12_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_12_1d)
    
    # Calculate 26-period EMA on daily close
    ema_26_1d = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_26_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_26_1d)
    
    # Calculate 12h ATR for stop loss
    tr_12h_1 = high - low
    tr_12h_2 = np.abs(high - np.roll(close, 1))
    tr_12h_3 = np.abs(low - np.roll(close, 1))
    tr_12h_1[0] = high[0] - low[0]
    tr_12h_2[0] = np.abs(high[0] - close[0])
    tr_12h_3[0] = np.abs(low[0] - close[0])
    tr_12h = np.maximum(tr_12h_1, np.maximum(tr_12h_2, tr_12h_3))
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need enough data for EMA26
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(ema_12_1d_aligned[i]) or 
            np.isnan(ema_26_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # MACD line (12 EMA - 26 EMA)
        macd_line = ema_12_1d_aligned[i] - ema_26_1d_aligned[i]
        
        # MACD signal line (9-period EMA of MACD line)
        # We'll approximate using previous values for signal line
        if i > start_idx:
            macd_signal = 0.9 * macd_signal_prev + 0.1 * macd_line
        else:
            macd_signal = macd_line  # Initialize
        macd_signal_prev = macd_signal
        
        # MACD histogram
        macd_hist = macd_line - macd_signal
        
        if position == 0:
            # Long entry: MACD histogram crosses above zero, with volume confirmation
            if macd_hist > 0 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: MACD histogram crosses below zero, with volume confirmation
            elif macd_hist < 0 and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: MACD histogram crosses below zero or ATR-based stop
            if macd_hist < 0 or close[i] < open_price[i] - 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: MACD histogram crosses above zero or ATR-based stop
            if macd_hist > 0 or close[i] > open_price[i] + 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_MACD_Histogram_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0