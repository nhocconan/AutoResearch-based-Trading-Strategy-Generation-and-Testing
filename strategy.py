#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load daily data once for HL2 and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # HL2 (median price) for 1d
    hl2_1d = (high_1d + low_1d) / 2
    
    # Calculate 20-period EMA of HL2 on daily
    hl2_series = pd.Series(hl2_1d)
    ema20_hl2_1d = hl2_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 14-period RSI on daily close
    close_series = pd.Series(close_1d)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    
    # Calculate 20-period volume average on daily
    volume_series = pd.Series(volume_1d)
    vol_avg_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 6h timeframe
    ema20_hl2_aligned = align_htf_to_ltf(prices, df_1d, ema20_hl2_1d)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any data is not ready
        if (np.isnan(ema20_hl2_aligned[i]) or 
            np.isnan(rsi_14_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = prices['volume'].iloc[i]
        ema20_hl2 = ema20_hl2_aligned[i]
        rsi_14 = rsi_14_aligned[i]
        vol_avg_20 = vol_avg_20_aligned[i]
        
        if position == 0:
            # Long: price above EMA20(HL2) + RSI > 50 + volume spike
            if price > ema20_hl2 and rsi_14 > 50 and vol > 1.5 * vol_avg_20:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA20(HL2) + RSI < 50 + volume spike
            elif price < ema20_hl2 and rsi_14 < 50 and vol > 1.5 * vol_avg_20:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back below/above EMA20(HL2)
            if position == 1 and price < ema20_hl2:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > ema20_hl2:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_EMA20_HL2_RSI14_Volume_Filter"
timeframe = "6h"
leverage = 1.0