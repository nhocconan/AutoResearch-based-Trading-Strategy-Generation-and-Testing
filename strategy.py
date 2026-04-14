#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Reversal with Weekly Trend Filter and Volume Confirmation
# Uses Camarilla pivot levels from 1d to identify reversal zones at R3/S3 and breakout continuation at R4/S4
# Weekly trend filter (1w EMA 20) ensures trades align with higher timeframe momentum
# Volume confirmation (>1.5x average volume) filters for institutional participation
# Designed to work in both bull and bear markets by trading reversals in ranging markets
# and breakouts in trending markets, aligned with weekly trend
# Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We focus on R3/S3 for reversals and R4/S4 for breakouts
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 6s timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA (20) for trend direction
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: weekly EMA direction
        above_weekly_ema = price > ema_1w_aligned[i]
        
        if position == 0:
            # Look for reversal at S3/R3 or breakout at S4/R4
            # Long conditions:
            # 1. Reversal: price rejects S3 (bounces off support) in uptrend
            # 2. Breakout: price breaks above R4 with volume
            long_reversal = (price > s3_aligned[i] and 
                           low[i] <= s3_aligned[i] * 1.002 and  # touched S3
                           above_weekly_ema)
            long_breakout = (price > r4_aligned[i] and 
                           vol > 1.5 * avg_vol[i])
            
            # Short conditions:
            # 1. Reversal: price rejects R3 (fails at resistance) in downtrend
            # 2. Breakout: price breaks below S4 with volume
            short_reversal = (price < r3_aligned[i] and 
                            high[i] >= r3_aligned[i] * 0.998 and  # touched R3
                            not above_weekly_ema)
            short_breakout = (price < s4_aligned[i] and 
                            vol > 1.5 * avg_vol[i])
            
            if long_reversal or long_breakout:
                position = 1
                signals[i] = position_size
            elif short_reversal or short_breakout:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
                
        elif position == 1:
            # Exit long: reversal at R3 or break below S3
            if (price < r3_aligned[i] and 
                high[i] >= r3_aligned[i] * 0.998):  # touched R3 from below
                position = 0
                signals[i] = 0.0
            elif price < s3_aligned[i]:  # broke below support
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: reversal at S3 or break above R3
            if (price > s3_aligned[i] and 
                low[i] <= s3_aligned[i] * 1.002):  # touched S3 from above
                position = 0
                signals[i] = 0.0
            elif price > r3_aligned[i]:  # broke above resistance
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Camarilla_Pivot_Reversal_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0