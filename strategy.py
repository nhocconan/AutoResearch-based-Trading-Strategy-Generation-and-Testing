#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot-based mean reversion with daily volume confirmation and 1d trend filter.
# Long when price touches S1/S2 AND volume > 1.5x daily average volume AND 1d close > 1d EMA34 (bullish trend)
# Short when price touches R1/R2 AND volume > 1.5x daily average volume AND 1d close < 1d EMA34 (bearish trend)
# Exit when price returns to pivot point (P)
# Uses Camarilla for mean-reversion levels, volume for confirmation, daily EMA34 for trend filter.
# Target: 20-30 trades/year per symbol.
name = "4h_Camarilla_Pivot_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1 = close_1d + range_1d * 1.1 / 12
    r2 = close_1d + range_1d * 1.1 / 6
    s1 = close_1d - range_1d * 1.1 / 12
    s2 = close_1d - range_1d * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 1d average volume for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema34_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34_val = ema34_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        r2_val = r2_aligned[i]
        s1_val = s1_aligned[i]
        s2_val = s2_aligned[i]
        
        # Trend filter: only trade in direction of 1d EMA34
        bullish_trend = close_1d[i] > ema34_val if i < len(close_1d) else False
        bearish_trend = close_1d[i] < ema34_val if i < len(close_1d) else False
        
        if position == 0:
            # Long entry: price touches S1 or S2 + volume spike + bullish trend
            if ((price <= s1_val or price <= s2_val) and 
                vol > 1.5 * vol_ma and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short entry: price touches R1 or R2 + volume spike + bearish trend
            elif ((price >= r1_val or price >= r2_val) and 
                  vol > 1.5 * vol_ma and bearish_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point
            if price >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point
            if price <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals