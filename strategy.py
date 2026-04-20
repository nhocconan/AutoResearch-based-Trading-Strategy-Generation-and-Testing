#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_RSI_Divergence_Trend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily and 4h data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    if len(df_1d) < 20 or len(df_4h) < 20:
        return np.zeros(n)
    
    # === Daily Trend Filter (1d EMA50) ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h RSI for divergence detection ===
    close_4h = df_4h['close'].values
    delta = pd.Series(close_4h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_values = rsi_4h.values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_values)
    
    # === 1h RSI for entry timing ===
    close_series = pd.Series(prices['close'])
    delta_1h = close_series.diff()
    gain_1h = delta_1h.where(delta_1h > 0, 0)
    loss_1h = -delta_1h.where(delta_1h < 0, 0)
    avg_gain_1h = pd.Series(gain_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1h = pd.Series(loss_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1h = avg_gain_1h / avg_loss_1h
    rsi_1h = 100 - (100 / (1 + rs_1h))
    rsi_1h_values = rsi_1h.values
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_1h_val = rsi_1h_values[i]
        rsi_4h_val = rsi_4h_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_1h_val) or np.isnan(rsi_4h_val) or 
            np.isnan(ema_50_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend + bullish RSI divergence (4h RSI making higher low while price makes lower low)
            # Simplified: RSI oversold recovery in uptrend
            if (close_val > ema_50_1d_val and 
                rsi_1h_val < 30 and 
                rsi_4h_val > 30 and 
                rsi_4h_val < 50):
                signals[i] = 0.20
                position = 1
            # Short: Downtrend + bearish RSI divergence (4h RSI making lower high while price makes higher high)
            # Simplified: RSI overbought rejection in downtrend
            elif (close_val < ema_50_1d_val and 
                  rsi_1h_val > 70 and 
                  rsi_4h_val < 70 and 
                  rsi_4h_val > 50):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or trend break
            if (rsi_1h_val > 70 or 
                close_val < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI oversold or trend break
            if (rsi_1h_val < 30 or 
                close_val > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals