#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 6d RSI(14)
    close_6h = prices['close'].values
    delta_6h = np.diff(close_6h, prepend=close_6h[0])
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    avg_gain_6h = pd.Series(gain_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_6h = pd.Series(loss_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_6h = avg_gain_6h / (avg_loss_6h + 1e-10)
    rsi_6h = 100 - (100 / (1 + rs_6h))
    
    # Volume ratio: current volume / 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi_6h[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_1d_val = rsi_1d_aligned[i]
        rsi_6h_val = rsi_6h[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: 1d RSI oversold (<30) + 6h RSI rising from oversold + volume confirmation
            if (rsi_1d_val < 30 and 
                rsi_6h_val < 35 and 
                rsi_6h_val > rsi_6h[i-1] and  # RSI rising
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: 1d RSI overbought (>70) + 6h RSI falling from overbought + volume confirmation
            elif (rsi_1d_val > 70 and 
                  rsi_6h_val > 65 and 
                  rsi_6h_val < rsi_6h[i-1] and  # RSI falling
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: RSI returns to neutral zone (40-60) or volume drops
            if position == 1 and (rsi_6h_val > 50 or vol_ratio_val < 0.8):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (rsi_6h_val < 50 or vol_ratio_val < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_RSIOversold_1dFilter_Volume"
timeframe = "6h"
leverage = 1.0