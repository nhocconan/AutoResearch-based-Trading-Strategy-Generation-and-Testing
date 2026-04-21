#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily RSI (14) ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = gain_ma / loss_ma
    rs = np.where(loss_ma == 0, 0, rs)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = rsi_14.values
    
    # === Daily SMA (50) ===
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align to 4h timeframe (use previous day's values)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    
    # === 4h Price ===
    close_4h = prices['close'].values
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(sma_50_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close_4h[i]
        rsi_val = rsi_14_aligned[i]
        sma_val = sma_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price above SMA50, RSI oversold (<30) with volume
            if (price_close > sma_val and 
                rsi_val < 30 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price below SMA50, RSI overbought (>70) with volume
            elif (price_close < sma_val and 
                  rsi_val > 70 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse conditions or volume drops
            if position == 1 and (price_close < sma_val or rsi_val > 70 or vol_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > sma_val or rsi_val < 30 or vol_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_RSI_SMA50_Volume_Filter"
timeframe = "4h"
leverage = 1.0