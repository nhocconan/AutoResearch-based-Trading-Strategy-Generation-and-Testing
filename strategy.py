#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w1d_Pivot_R3S3_Fade_With_Trend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 5 or len(df_1d) < 5:
        return np.zeros(n)
    
    # === 1d: Calculate Camarilla pivot levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first period uses current
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Pivot point and ranges
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels: R3, R4, S3, S4
    r3 = pivot + (range_val * 1.1 / 4)
    r4 = pivot + (range_val * 1.1 / 2)
    s3 = pivot - (range_val * 1.1 / 4)
    s4 = pivot - (range_val * 1.1 / 2)
    
    # Align to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 1w: Trend filter (price above/below weekly EMA20) ===
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Session filter: 00-23 UTC (all hours for 6h)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        r3_val = r3_6h[i]
        r4_val = r4_6h[i]
        s3_val = s3_6h[i]
        s4_val = s4_6h[i]
        ema_wk = ema20_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: Fade at S3/S4 in weekly uptrend
            if (close_val <= s3_val and          # Price at or below S3 (strong support)
                ema_wk > close_val and           # Weekly uptrend (price below EMA = pullback in uptrend)
                vol_ratio_val > 1.5):            # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Fade at R3/R4 in weekly downtrend
            elif (close_val >= r3_val and        # Price at or above R3 (strong resistance)
                  ema_wk < close_val and         # Weekly downtrend (price above EMA = rally in downtrend)
                  vol_ratio_val > 1.5):          # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: reach pivot or trend breakdown
            if (close_val >= pivot[i] or         # Reached pivot level (take profit)
                ema_wk < close_val):             # Weekly trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reach pivot or trend reversal
            if (close_val <= pivot[i] or         # Reached pivot level (take profit)
                ema_wk > close_val):             # Weekly trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals