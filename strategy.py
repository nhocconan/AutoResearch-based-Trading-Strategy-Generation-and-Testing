#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot levels + volume confirmation + choppiness regime filter
# Long when price touches Camarilla S3 level with volume confirmation in choppy market (mean reversion)
# Short when price touches Camarilla R3 level with volume confirmation in choppy market
# Uses discrete position sizing 0.25 to target ~25-40 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion in chop, avoids trending markets via chop filter

name = "4h_12h_camarilla_pivot_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S3 = C - (Range * 1.100/4)
    # S2 = C - (Range * 1.100/2)
    # S1 = C - (Range * 1.100/6)
    # R1 = C + (Range * 1.100/6)
    # R2 = C + (Range * 1.100/2)
    # R3 = C + (Range * 1.100/4)
    
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    s3_12h = close_12h - (range_12h * 1.100 / 4.0)
    s2_12h = close_12h - (range_12h * 1.100 / 2.0)
    s1_12h = close_12h - (range_12h * 1.100 / 6.0)
    r1_12h = close_12h + (range_12h * 1.100 / 6.0)
    r2_12h = close_12h + (range_12h * 1.100 / 2.0)
    r3_12h = close_12h + (range_12h * 1.100 / 4.0)
    
    # Align 12h Camarilla levels to 4h timeframe (wait for completed 12h bar)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    
    # Calculate 4h average volume (20-period) for volume confirmation
    vol_s = pd.Series(volume)
    avg_vol_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Choppiness Index (14-period) for regime filter
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = true_range(high, low, prev_close)
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr14) / (max(high14) - min(low14))) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = highest_high_14 - lowest_low_14
    chop = np.where(denominator > 0, 
                    100 * np.log10(sum_tr_14 / denominator) / np.log10(14), 
                    50.0)  # neutral when denominator is 0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(avg_vol_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.8x average 4h volume (20-period)
        volume_confirmed = volume[i] > 1.8 * avg_vol_20[i]
        
        # Chop regime filter: only trade in choppy market (Chop > 61.8 = ranging)
        chop_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit long if price moves above S2 (take profit) or below S3 (stop)
            if close[i] > s2_aligned[i] or close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price moves below R2 (take profit) or above R3 (stop)
            if close[i] < r2_aligned[i] or close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion strategy: enter at extreme Camarilla levels with volume confirmation in chop
            if close[i] <= s3_aligned[i] and volume_confirmed and chop_filter:
                position = 1
                signals[i] = 0.25
            elif close[i] >= r3_aligned[i] and volume_confirmed and chop_filter:
                position = -1
                signals[i] = -0.25
    
    return signals