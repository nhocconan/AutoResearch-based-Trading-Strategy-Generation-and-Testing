#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot levels with volume confirmation and choppiness regime filter
# Camarilla pivots from 12h provide intraday support/resistance levels that work in ranging markets
# Volume spike confirmation (current 4h volume > 2.0x 20-period average) filters false breakouts
# Choppiness regime filter (CHOP(14) > 61.8) ensures we only trade in ranging conditions
# Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_12h_camarilla_volume_chop_v1"
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
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    #          S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # We use the previous completed 12h bar for calculation (lookback 1)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    
    # Handle first bar (no previous data)
    prev_high[0] = high_12h[0]
    prev_low[0] = low_12h[0]
    prev_close[0] = close_12h[0]
    
    range_12h = prev_high - prev_low
    camarilla_r4 = prev_close + range_12h * 1.1 / 2.0
    camarilla_r3 = prev_close + range_12h * 1.1 / 4.0
    camarilla_r2 = prev_close + range_12h * 1.1 / 6.0
    camarilla_r1 = prev_close + range_12h * 1.1 / 12.0
    camarilla_s1 = prev_close - range_12h * 1.1 / 12.0
    camarilla_s2 = prev_close - range_12h * 1.1 / 6.0
    camarilla_s3 = prev_close - range_12h * 1.1 / 4.0
    camarilla_s4 = prev_close - range_12h * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    r2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r2)
    r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    s2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s2)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute choppiness index (14-period) for 4h
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log10(n))) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR_sum / (highest_high - lowest_low)) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = hh_14 - ll_14
    # Avoid division by zero
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_raw = 100 * np.log10(atr_sum / chop_denom) / np.log10(14)
    chop_14 = chop_raw  # Already in 0-100 range
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop_14[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2.0x average 4h volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Choppiness regime filter: only trade when CHOP > 61.8 (ranging market)
        chop_filter = chop_14[i] > 61.8
        
        if not volume_confirmed or not chop_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size for consistency (discrete levels to minimize fee churn)
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price reaches Camarilla R3 (take profit) or retreats below S1 (stop)
            if close[i] >= r3_aligned[i] or close[i] <= s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price reaches Camarilla S3 (take profit) or retreats above R1 (stop)
            if close[i] <= s3_aligned[i] or close[i] >= r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Mean reversion trading at extreme Camarilla levels with volume confirmation
            # Long at S4 with volume confirmation, Short at R4 with volume confirmation
            if volume_confirmed:
                if close[i] <= s4_aligned[i]:
                    position = 1
                    signals[i] = position_size
                elif close[i] >= r4_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals