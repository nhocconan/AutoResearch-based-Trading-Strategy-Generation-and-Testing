#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with volume confirmation and daily trend filter.
# Long when price breaks above R1 AND volume > 1.3x daily average volume AND price > daily EMA50 (bullish trend)
# Short when price breaks below S1 AND volume > 1.3x daily average volume AND price < daily EMA50 (bearish trend)
# Exit when price returns to daily pivot point (central level)
# Uses Camarilla levels for precise intraday S/R, volume for breakout confirmation, daily EMA for trend filter.
# Target: 20-30 trades/year per symbol.
name = "4h_Camarilla_R1S1_Volume_EMA50"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (R1, S1, pivot) from previous day
    # Using previous day's high, low, close
    phigh = df_1d['high'].shift(1).values
    plow = df_1d['low'].shift(1).values
    pclose = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    pivot = (phigh + plow + pclose) / 3
    range_ = phigh - plow
    r1 = pclose + range_ * 1.1 / 12
    s1 = pclose - range_ * 1.1 / 12
    
    # Calculate daily EMA50 for trend filter
    ema50 = pd.Series(pclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily average volume (20-day)
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        ema50_val = ema50_aligned[i]
        vol_ma = vol_ma_aligned[i]
        vol = volume[i]
        
        # Volume confirmation: above average
        vol_confirm = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long entry: break above R1 + volume + bullish trend (price > EMA50)
            if price > r1_val and vol_confirm and price > ema50_val:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 + volume + bearish trend (price < EMA50)
            elif price < s1_val and vol_confirm and price < ema50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot
            if price <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot
            if price >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals