#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Weekly Williams %R (14-period) for oversold/overbought ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate highest high and lowest low over 14 weeks
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close_1w) / (highest_high - lowest_low)
    
    # Align Williams %R to daily timeframe
    willr_aligned = align_htf_to_ltf(prices, df_1w, willr)
    
    # === Daily RSI (14-period) for momentum confirmation ===
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to daily timeframe (already daily, but keep for consistency)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === Daily volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(willr_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        willr_val = willr_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long when weekly Williams %R shows oversold (< -80) and daily RSI confirms momentum (> 50) with volume
            if (willr_val < -80 and  # Weekly oversold
                rsi_val > 50 and     # Daily bullish momentum
                vol_ratio_val > 1.5): # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Enter short when weekly Williams %R shows overbought (> -20) and daily RSI confirms weakness (< 50) with volume
            elif (willr_val > -20 and   # Weekly overbought
                  rsi_val < 50 and      # Daily bearish momentum
                  vol_ratio_val > 1.5): # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when weekly Williams %R moves out of extreme zone or RSI reverses
            if position == 1 and (willr_val > -50 or rsi_val < 40):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (willr_val < -50 or rsi_val > 60):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Williams_RSI_Volume_Strategy"
timeframe = "1d"
leverage = 1.0