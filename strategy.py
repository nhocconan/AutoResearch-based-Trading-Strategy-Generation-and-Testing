#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Reversal with Daily Volume Spike
# Trade reversals at weekly Camarilla pivot levels (R3/S3, R4/S4) when price shows exhaustion
# Daily volume spike (>2x average) confirms institutional interest at these key levels
# Works in bull markets (buy R3/S3 bounce in uptrend) and bear markets (sell R3/S4 rejection in downtrend)
# Weekly pivots provide structure; volume confirms validity; 6h timeframe avoids noise
# Target: 15-25 trades/year (60-100 total over 4 years)

name = "6h_WeeklyPivot_Reversal_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for weekly and daily calculations
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivots from previous week
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # Using previous week's OHLC to avoid look-ahead
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    wk_high_prev = np.roll(wk_high, 1)
    wk_low_prev = np.roll(wk_low, 1)
    wk_close_prev = np.roll(wk_close, 1)
    wk_high_prev[0] = np.nan
    wk_low_prev[0] = np.nan
    wk_close_prev[0] = np.nan
    
    wk_range = wk_high_prev - wk_low_prev
    camarilla_r4 = wk_close_prev + (wk_range * 1.1 / 2)
    camarilla_r3 = wk_close_prev + (wk_range * 1.1 / 4)
    camarilla_s3 = wk_close_prev - (wk_range * 1.1 / 4)
    camarilla_s4 = wk_close_prev - (wk_range * 1.1 / 2)
    
    # Align weekly pivots to 6h timeframe (wait for weekly bar to close)
    r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Calculate daily volume average for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20_aligned[i]
        
        # Volume confirmation: current volume > 2x daily average
        volume_spike = curr_volume > (2.0 * curr_vol_ma)
        
        if position == 0:  # Flat - look for reversal entries
            # Bullish reversal: price tests and holds above S3/S4 with volume spike
            if volume_spike:
                # Test S3 level (strong support)
                if curr_low <= s3_aligned[i] * 1.002 and curr_close > s3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Test S4 level (extreme support)
                elif curr_low <= s4_aligned[i] * 1.005 and curr_close > s4_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Bearish reversal: price tests and holds below R3/R4 with volume spike
            elif volume_spike:
                # Test R3 level (strong resistance)
                if curr_high >= r3_aligned[i] * 0.998 and curr_close < r3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                # Test R4 level (extreme resistance)
                elif curr_high >= r4_aligned[i] * 0.995 and curr_close < r4_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price reaches R3 (profit target) or breaks below S3 (stop)
            if curr_close >= r3_aligned[i] or curr_close < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price reaches S3 (profit target) or breaks above R3 (stop)
            if curr_close <= s3_aligned[i] or curr_close > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals