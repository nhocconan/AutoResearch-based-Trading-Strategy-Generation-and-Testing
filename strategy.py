#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Daily RSI for regime filter (14-period)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # RSI calculation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 6h price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h EMA50 for trend
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # 6h volume and volume SMA20 for confirmation
    volume = prices['volume'].values
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(vol_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        ema50_val = ema50[i]
        rsi_val = rsi_1d_aligned[i]
        vol_sma20_val = vol_sma20[i]
        
        # Volume confirmation: volume above 20-period average
        vol_confirmed = vol > vol_sma20_val
        
        if position == 0 and vol_confirmed:
            # Long: price above EMA50 + daily RSI in bullish range (50-70)
            if price > ema50_val and 50 <= rsi_val <= 70:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below EMA50 + daily RSI in bearish range (30-50)
            elif price < ema50_val and 30 <= rsi_val <= 50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit: price crosses back below/above EMA50 or RSI exits range
            if position == 1:
                exit_cond = (price < ema50_val) or (rsi_val < 50)
            else:  # position == -1
                exit_cond = (price > ema50_val) or (rsi_val > 50)
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_EMA50_DailyRSI_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0