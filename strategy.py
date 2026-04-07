# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_camarilla_pivot_1w_trend_volume_v1
Hypothesis: Weekly Camarilla pivot levels provide strong institutional support/resistance.
Price reverses at S3/R3 and breaks out at S4/R4 with volume confirmation.
Daily EMA filter ensures alignment with weekly trend.
Designed for 1D timeframe to capture multi-day moves with low trade frequency.
Works in bull/bear by following weekly trend direction.
Target: 10-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Weekly data for Camarilla pivot and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous week
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Calculate pivot levels
    pp = (prev_high + prev_low + prev_close) / 3
    r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    r2 = prev_close + ((prev_high - prev_low) * 1.1 / 6)
    r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    s2 = prev_close - ((prev_high - prev_low) * 1.1 / 6)
    s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align pivot levels to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Weekly EMA filter
    weekly_ema = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(weekly_ema_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter
        above_weekly_ema = close[i] > weekly_ema_aligned[i]
        below_weekly_ema = close[i] < weekly_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price below S3 or loss of weekly uptrend
            if close[i] < s3_aligned[i] or not above_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price above R3 or loss of weekly downtrend
            if close[i] > r3_aligned[i] or not below_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above R4 with volume and above weekly EMA
            if (close[i] > r4_aligned[i] and 
                vol_confirm and 
                above_weekly_ema):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S4 with volume and below weekly EMA
            elif (close[i] < s4_aligned[i] and 
                  vol_confirm and 
                  below_weekly_ema):
                position = -1
                signals[i] = -0.25
            # Mean reversion at S3/R3 with volume and trend alignment
            elif (close[i] <= s3_aligned[i] and 
                  vol_confirm and 
                  above_weekly_ema):
                # Buy at S3 support in uptrend
                position = 1
                signals[i] = 0.25
            elif (close[i] >= r3_aligned[i] and 
                  vol_confirm and 
                  below_weekly_ema):
                # Sell at R3 resistance in downtrend
                position = -1
                signals[i] = -0.25
    
    return signals