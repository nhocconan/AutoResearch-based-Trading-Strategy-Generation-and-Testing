#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot levels + volume confirmation + choppiness regime filter
# Camarilla pivots from 12h provide key support/resistance levels that work in both bull/bear markets
# Long when price touches Camarilla S3 with volume confirmation and chop > 61.8 (range)
# Short when price touches Camarilla R3 with volume confirmation and chop > 61.8 (range)
# Uses discrete position sizing 0.25 to target ~25-40 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion at extreme pivots during range regimes, avoids trending markets

name = "4h_12h_camarilla_pivot_volume_chop_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Camarilla: S3 = C - (H-L)*1.1/4, R3 = C + (H-L)*1.1/4
    camarilla_s3 = close_12h - (high_12h - low_12h) * 1.1 / 4
    camarilla_r3 = close_12h + (high_12h - low_12h) * 1.1 / 4
    
    # Calculate 12h average volume (20-period)
    vol_s_12h = pd.Series(df_12h['volume'].values)
    avg_vol_12h = vol_s_12h.rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    avg_vol_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_vol_12h)
    
    # Calculate 4h Choppiness Index (14-period) for regime filter
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = 0
    tr = true_range(high, low, prev_close)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(TR14)/(ATR14*14)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    
    # Pre-compute volume confirmation: current 4h volume > 1.5x average 4h volume (20-period)
    vol_s_4h = pd.Series(volume)
    avg_vol_4h = vol_s_4h.rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * avg_vol_4h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(avg_vol_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price moves above Camarilla S3 (mean reversion complete)
            if close[i] > camarilla_s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price moves below Camarilla R3 (mean reversion complete)
            if close[i] < camarilla_r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion strategy: enter at extreme Camarilla levels during range regime
            if (close[i] <= camarilla_s3_aligned[i] and 
                chop[i] > 61.8 and 
                volume_confirmed[i]):
                position = 1
                signals[i] = 0.25
            elif (close[i] >= camarilla_r3_aligned[i] and 
                  chop[i] > 61.8 and 
                  volume_confirmed[i]):
                position = -1
                signals[i] = -0.25
    
    return signals