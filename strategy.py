#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_RSI_Pullback_MultiTimeframe"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss != 0, avg_loss, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 6h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 6h RSI(14) for entry timing
    close_6h = prices['close'].values
    delta_6h = np.diff(close_6h, prepend=close_6h[0])
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    avg_gain_6h = pd.Series(gain_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_6h = pd.Series(loss_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_6h = avg_gain_6h / np.where(avg_loss_6h != 0, avg_loss_6h, np.nan)
    rsi_6h = 100 - (100 / (1 + rs_6h))
    
    # 6h EMA(50) for trend filter
    ema50_6h = pd.Series(close_6h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 6h volume ratio (current vs 20-period average)
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA and RSI warmup
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_1d_val = rsi_1d_aligned[i]
        rsi_6h_val = rsi_6h[i]
        ema50_6h_val = ema50_6h[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_1d_val) or np.isnan(rsi_6h_val) or np.isnan(ema50_6h_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d RSI oversold (<30) + 6h RSI pulling back from oversold (<40) + price above EMA50 + volume confirmation
            if (rsi_1d_val < 30 and rsi_6h_val < 40 and 
                close_val > ema50_6h_val and vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: 1d RSI overbought (>70) + 6h RSI pulling back from overbought (>60) + price below EMA50 + volume confirmation
            elif (rsi_1d_val > 70 and rsi_6h_val > 60 and 
                  close_val < ema50_6h_val and vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: 1d RSI returns to neutral (>50) or price breaks below EMA50
            if rsi_1d_val > 50 or close_val < ema50_6h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: 1d RSI returns to neutral (<50) or price breaks above EMA50
            if rsi_1d_val < 50 or close_val > ema50_6h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals