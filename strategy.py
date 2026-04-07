#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v2
Hypothesis: Camarilla pivot levels from daily timeframe provide institutional-grade support/resistance.
Price tends to reverse at S3/R3 and breakout at S4/R4 with volume confirmation.
EMA filter ensures alignment with daily trend to avoid counter-trend trades.
Designed for 6H timeframe to reduce trade frequency and fee drag while capturing significant moves.
Works in both bull/bear markets by following daily trend direction.
Target: 15-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R4 = C + ((H - L) * 1.1 / 2)
    # R3 = C + ((H - L) * 1.1 / 4)
    # R2 = C + ((H - L) * 1.1 / 6)
    # R1 = C + ((H - L) * 1.1 / 12)
    # S1 = C - ((H - L) * 1.1 / 12)
    # S2 = C - ((H - L) * 1.1 / 6)
    # S3 = C - ((H - L) * 1.1 / 4)
    # S4 = C - ((H - L) * 1.1 / 2)
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
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
    
    # Align pivot levels to 6H timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily EMA filter
    daily_ema = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(daily_ema_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter
        above_daily_ema = close[i] > daily_ema_aligned[i]
        below_daily_ema = close[i] < daily_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price below S3 or loss of daily uptrend
            if close[i] < s3_aligned[i] or not above_daily_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price above R3 or loss of daily downtrend
            if close[i] > r3_aligned[i] or not below_daily_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above R4 with volume and above daily EMA
            if (close[i] > r4_aligned[i] and 
                vol_confirm and 
                above_daily_ema):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S4 with volume and below daily EMA
            elif (close[i] < s4_aligned[i] and 
                  vol_confirm and 
                  below_daily_ema):
                position = -1
                signals[i] = -0.25
            # Mean reversion at S3/R3 with volume and trend alignment
            elif (close[i] <= s3_aligned[i] and 
                  vol_confirm and 
                  above_daily_ema):
                # Buy at S3 support in uptrend
                position = 1
                signals[i] = 0.25
            elif (close[i] >= r3_aligned[i] and 
                  vol_confirm and 
                  below_daily_ema):
                # Sell at R3 resistance in downtrend
                position = -1
                signals[i] = -0.25
    
    return signals