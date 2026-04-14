#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate RSI (14-period) on 12h data
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Calculate ATR (14-period) on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_prev = np.roll(close_12h, 1)
    close_12h_prev[0] = close_12h[0]
    
    tr = np.maximum(high_12h - low_12h, 
                    np.maximum(np.abs(high_12h - close_12h_prev), 
                               np.abs(low_12h - close_12h_prev)))
    
    atr_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h EMA (50-period) on close
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6-period average volume (4h periods in a day for 12h TF)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Get aligned indicators
        rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)[i]
        atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)[i]
        ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)[i]
        vol_ma_6_val = vol_ma_6[i]  # already LTF
        
        # Check for NaN values
        if (np.isnan(rsi_12h_aligned) or np.isnan(atr_12h_aligned) or 
            np.isnan(ema_50_12h_aligned) or np.isnan(vol_ma_6_val)):
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma_6_val
        
        if position == 0:  # No position - look for entries
            if volume_confirm:
                # Long: RSI < 30 (oversold) and price above EMA50 (bullish bias)
                if rsi_12h_aligned < 30 and close[i] > ema_50_12h_aligned:
                    position = 1
                    signals[i] = position_size
                # Short: RSI > 70 (overbought) and price below EMA50 (bearish bias)
                elif rsi_12h_aligned > 70 and close[i] < ema_50_12h_aligned:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when RSI > 70 (overbought)
            if rsi_12h_aligned > 70:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when RSI < 30 (oversold)
            if rsi_12h_aligned < 30:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_RSI12h_OverboughtOversold_EMA50_Volume1.5x_v1"
timeframe = "6h"
leverage = 1.0