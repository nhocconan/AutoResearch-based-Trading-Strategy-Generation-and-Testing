#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v1
Hypothesis: On 6-hour timeframe, use daily Camarilla pivot levels (R3, S3) for mean reversion entries and (R4, S4) for breakout continuation, filtered by 1-day EMA trend and volume confirmation. This strategy exploits intraday reversions to daily mean in ranging markets and breakouts in trending markets, with volume ensuring institutional participation. Designed for 50-150 total trades over 4 years (~12-37/year) to minimize fee drag while performing in both bull and bear regimes.
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
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and EMA trend
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA(50) to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Calculate Camarilla levels
    # R4 = close + ((high - low) * 1.5)
    # R3 = close + ((high - low) * 1.25)
    # S3 = close - ((high - low) * 1.25)
    # S4 = close - ((high - low) * 1.5)
    range_hl = prev_high - prev_low
    r4 = prev_close + (range_hl * 1.5)
    r3 = prev_close + (range_hl * 1.25)
    s3 = prev_close - (range_hl * 1.25)
    s4 = prev_close - (range_hl * 1.5)
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter: 24-period average on 6h timeframe (equivalent to ~6 days)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(24, 50), n):
        # Skip if data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R4 (take profit) or breaks below S3 (stop)
            if high[i] >= r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif low[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S4 (take profit) or breaks above R3 (stop)
            if low[i] <= s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif high[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Mean reversion: fade at R3/S3 when price is overextended
                # Long: price touches or goes below S3 but not S4, with upward EMA bias
                if low[i] <= s3_aligned[i] and low[i] > s4_aligned[i] and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches or goes above R3 but not R4, with downward EMA bias
                elif high[i] >= r3_aligned[i] and high[i] < r4_aligned[i] and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
                # Breakout continuation: break through R4/S4 with strong trend
                # Long: break above R4 with strong upward EMA
                elif high[i] >= r4_aligned[i] and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-3]:  # Stronger trend
                    position = 1
                    signals[i] = 0.25
                # Short: break below S4 with strong downward EMA
                elif low[i] <= s4_aligned[i] and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-3]:  # Stronger trend
                    position = -1
                    signals[i] = -0.25
    
    return signals