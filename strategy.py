#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Camarilla_R1S1_Breakout_Volume_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Points (previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Set first values to avoid look-ahead
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Classic pivot (same for Camarilla)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels - Focus on R1/S1 for breakouts
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume Confirmation (6h) ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Trend Filter (6h EMA34) ===
    close = prices['close'].values
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema34[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(ema_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Breakout at R1/S1 with volume confirmation and price above/below EMA34
            if close_val > r1_val and vol_ratio_val > 2.0 and close_val > ema_val:
                # Break above R1 with upward trend
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif close_val < s1_val and vol_ratio_val > 2.0 and close_val < ema_val:
                # Break below S1 with downward trend
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: stop loss or return to S1
            if close_val <= entry_price - 1.5 * (prices['high'].iloc[i] - prices['low'].iloc[i]):
                # Stop loss hit
                signals[i] = 0.0
                position = 0
            elif close_val < s1_val:
                # Return to S1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or return to S1
            if close_val >= entry_price + 1.5 * (prices['high'].iloc[i] - prices['low'].iloc[i]):
                # Stop loss hit
                signals[i] = 0.0
                position = 0
            elif close_val > s1_val:
                # Return to S1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals