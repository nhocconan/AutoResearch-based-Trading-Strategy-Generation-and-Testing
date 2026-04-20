#!/usr/bin/env python3
"""
6h_1d_RSI_Extremes_TrendFollow_v1
Concept: RSI extremes on 1d timeframe combined with 6h EMA trend filter and volume confirmation.
- Long when 1d RSI < 30 (oversold) and price above 6h EMA50 with volume confirmation
- Short when 1d RSI > 70 (overbought) and price below 6h EMA50 with volume confirmation
- Exit when RSI returns to neutral zone (40-60) or trend changes
- Works in bull/bear: RSI extremes capture reversals, EMA filter avoids counter-trend trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_RSI_Extremes_TrendFollow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # === Calculate daily RSI(14) ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:13] = np.nan  # Not enough data
    
    # Align RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 6h: EMA50 trend filter ===
    close = prices['close'].values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA50
    
    for i in range(start_idx, n):
        # Get values
        rsi_val = rsi_1d_aligned[i]
        ema50_val = ema50[i]
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(ema50_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + price above EMA50 + volume confirmation
            rsi_oversold = rsi_val < 30
            uptrend = close_val > ema50_val
            vol_confirm = vol_ratio_val > 1.5  # Volume significantly above average
            
            if rsi_oversold and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + price below EMA50 + volume confirmation
            elif rsi_val > 70 and close_val < ema50_val and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (>=40) or trend breaks
            if rsi_val >= 40 or close_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (<=60) or trend breaks
            if rsi_val <= 60 or close_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals