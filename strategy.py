#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (from 1d) + volume confirmation + chop regime filter
# Camarilla levels from daily data provide intraday support/resistance for breakout validation
# Volume confirmation ensures breakout authenticity with institutional participation
# Choppiness index regime filter avoids whipsaws in ranging markets (CHOP > 61.8 = range, < 38.2 = trend)
# Works in bull/bear: Camarilla adapts to daily structure, chop filter improves bear market performance
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25-0.30

name = "4h_1d_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from daily OHLC (using prior day's data)
    # Camarilla uses previous day's range to calculate intraday levels
    # Pivot = (Prior Day High + Prior Day Low + Prior Day Close) / 3
    # R4 = Prior Day Close + ((Prior Day High - Prior Day Low) * 1.1 / 2)
    # R3 = Prior Day Close + ((Prior Day High - Prior Day Low) * 1.1 / 4)
    # R2 = Prior Day Close + ((Prior Day High - Prior Day Low) * 1.1 / 6)
    # R1 = Prior Day Close + ((Prior Day High - Prior Day Low) * 1.1 / 12)
    # S1 = Prior Day Close - ((Prior Day High - Prior Day Low) * 1.1 / 12)
    # S2 = Prior Day Close - ((Prior Day High - Prior Day Low) * 1.1 / 6)
    # S3 = Prior Day Close - ((Prior Day High - Prior Day Low) * 1.1 / 4)
    # S4 = Prior Day Close - ((Prior Day High - Prior Day Low) * 1.1 / 2)
    
    # Shift by 1 to use prior day's data (completed day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r4 = prev_close + (range_hl * 1.1 / 2)
    r3 = prev_close + (range_hl * 1.1 / 4)
    r2 = prev_close + (range_hl * 1.1 / 6)
    r1 = prev_close + (range_hl * 1.1 / 12)
    s1 = prev_close - (range_hl * 1.1 / 12)
    s2 = prev_close - (range_hl * 1.1 / 6)
    s3 = prev_close - (range_hl * 1.1 / 4)
    s4 = prev_close - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 4-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 4:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-4:i])
    
    # Calculate Choppiness Index on 4h data (14-period)
    chop = np.full(n, np.nan)
    for i in range(n):
        if i < 14:
            chop[i] = np.nan
        else:
            # True Range calculation
            tr1 = high[i-13:i+1] - low[i-13:i+1]
            tr2 = np.abs(high[i-13:i+1] - np.roll(close[i-13:i+1], 1))
            tr3 = np.abs(low[i-13:i+1] - np.roll(close[i-13:i+1], 1))
            # Handle first element of roll
            tr2[0] = np.abs(high[i-13] - close[i-13])
            tr3[0] = np.abs(low[i-13] - close[i-13])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(tr)
            
            # Choppiness Index formula
            sum_tr = np.sum(tr)
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if highest_high != lowest_low:
                chop[i] = 100 * np.log10(sum_tr / atr) / np.log10(14)
            else:
                chop[i] = 50.0  # Neutral when no range
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(avg_volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 4-period average
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        # Chop regime: trend when CHOP < 40, range when CHOP > 62
        is_trending = chop[i] < 40.0
        
        if position == 1:  # Long position
            # Exit: price < S1 (stoploss) OR chop becomes too high (choppy market)
            if close[i] < s1_aligned[i] or chop[i] > 62.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > R1 (stoploss) OR chop becomes too high (choppy market)
            if close[i] > r1_aligned[i] or chop[i] > 62.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Camarilla breakout + chop filter
            if volume_confirmed and is_trending:
                # Long entry: price > R1 AND price > R2 (strong breakout above resistance)
                if close[i] > r1_aligned[i] and close[i] > r2_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < S1 AND price < S2 (strong breakout below support)
                elif close[i] < s1_aligned[i] and close[i] < s2_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals