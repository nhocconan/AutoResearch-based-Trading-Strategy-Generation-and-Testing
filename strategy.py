#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v1
Hypothesis: On 6h timeframe, use 1d Camarilla pivot levels for mean reversion at R3/S3 and breakout continuation at R4/S4, with 1d EMA for trend filter and volume confirmation for institutional participation. Enter long at S3 bounce in uptrend with volume, short at R3 rejection in downtrend with volume. Enter long on R4 breakout in uptrend with volume, short on S4 breakdown in downtrend with volume. This strategy captures both mean reversion in ranges and breakout trends, works in bull/bear via trend filter, and limits trades via strict pivot levels and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot, EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r1_1d = pivot_1d + (range_1d * 1.0833 / 12)
    r2_1d = pivot_1d + (range_1d * 1.0833 / 6)
    r3_1d = pivot_1d + (range_1d * 1.0833 / 4)
    r4_1d = pivot_1d + (range_1d * 1.0833 / 2)
    
    # Support levels
    s1_1d = pivot_1d - (range_1d * 1.0833 / 12)
    s2_1d = pivot_1d - (range_1d * 1.0833 / 6)
    s3_1d = pivot_1d - (range_1d * 1.0833 / 4)
    s4_1d = pivot_1d - (range_1d * 1.0833 / 2)
    
    # 1d EMA for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align indicators to 6h timeframe
    r3_1d_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_6h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_6h = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema_1d_6h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r3_1d_6h[i]) or np.isnan(r4_1d_6h[i]) or
            np.isnan(s3_1d_6h[i]) or np.isnan(s4_1d_6h[i]) or
            np.isnan(ema_1d_6h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend direction from EMA
        uptrend = close[i] > ema_1d_6h[i]
        downtrend = close[i] < ema_1d_6h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price reaches R4 (take profit)
            if close[i] >= r4_1d_6h[i]:
                exit_long = True
            # Exit if price breaks below S3 (stop/reversal)
            elif close[i] <= s3_1d_6h[i]:
                exit_long = True
            # Exit if trend turns down and price below pivot
            elif downtrend and close[i] < pivot_1d[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price reaches S4 (take profit)
            if close[i] <= s4_1d_6h[i]:
                exit_short = True
            # Exit if price breaks above R3 (stop/reversal)
            elif close[i] >= r3_1d_6h[i]:
                exit_short = True
            # Exit if trend turns up and price above pivot
            elif uptrend and close[i] > pivot_1d[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry at S3 bounce (mean reversion)
            long_entry_s3 = False
            if close[i] > s3_1d_6h[i] and close[i-1] <= s3_1d_6h[i-1]:
                if uptrend and vol_confirm:
                    long_entry_s3 = True
            
            # Long entry on R4 breakout (continuation)
            long_entry_r4 = False
            if close[i] > r4_1d_6h[i] and close[i-1] <= r4_1d_6h[i-1]:
                if uptrend and vol_confirm:
                    long_entry_r4 = True
            
            # Short entry at R3 rejection (mean reversion)
            short_entry_r3 = False
            if close[i] < r3_1d_6h[i] and close[i-1] >= r3_1d_6h[i-1]:
                if downtrend and vol_confirm:
                    short_entry_r3 = True
            
            # Short entry on S4 breakdown (continuation)
            short_entry_s4 = False
            if close[i] < s4_1d_6h[i] and close[i-1] >= s4_1d_6h[i-1]:
                if downtrend and vol_confirm:
                    short_entry_s4 = True
            
            if long_entry_s3 or long_entry_r4:
                position = 1
                signals[i] = 0.25
            elif short_entry_r3 or short_entry_s4:
                position = -1
                signals[i] = -0.25
    
    return signals