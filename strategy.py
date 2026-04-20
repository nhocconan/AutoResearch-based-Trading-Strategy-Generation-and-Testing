#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_RSI20_Breakout_Volume_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily RSI(20) - overbought/oversold levels ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 19 + gain[i]) / 20
        avg_loss[i] = (avg_loss[i-1] * 19 + loss[i]) / 20
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi_20 = 100 - (100 / (1 + rs))
    
    # RSI levels: oversold < 20, overbought > 80
    rsi_oversold = 20
    rsi_overbought = 80
    
    # Align RSI levels to 4h
    oversold_aligned = align_htf_to_ltf(prices, df_1d, rsi_oversold * np.ones_like(rsi_20))
    overbought_aligned = align_htf_to_ltf(prices, df_1d, rsi_overbought * np.ones_like(rsi_20))
    
    # === 4h RSI(14) for entry timing ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # === Volume filter ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        rsi_14_val = rsi_14[i]
        vol_ratio_val = vol_ratio[i]
        oversold_val = oversold_aligned[i]
        overbought_val = overbought_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_14_val) or np.isnan(vol_ratio_val) or 
            np.isnan(oversold_val) or np.isnan(overbought_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI14 crosses above 20 from below with volume confirmation
            if (rsi_14_val > 20 and rsi_14[i-1] <= 20 and 
                vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: RSI14 crosses below 80 from above with volume confirmation
            elif (rsi_14_val < 80 and rsi_14[i-1] >= 80 and 
                  vol_ratio_val > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI14 crosses below 50
            if rsi_14_val < 50 and rsi_14[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI14 crosses above 50
            if rsi_14_val > 50 and rsi_14[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals