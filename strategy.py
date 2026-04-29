#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Camarilla Pivot Breakout with Daily Volume Spike and ATR Filter
# Uses weekly Camarilla pivot levels (R3/S3) as strong support/resistance from 1w timeframe
# Breakouts above weekly R3 or below weekly S3 with daily volume confirmation (>2x 20-period average)
# and ATR(14) > ATR(50) filter to ensure sufficient volatility for follow-through
# Works in both bull and bear markets by capturing strong momentum moves from key weekly levels
# Target: 12-25 trades/year (50-100 total over 4 years)

name = "6h_WeeklyCamarilla_R3S3_Breakout_DailyVolSpike_ATRFilter_v1"
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
    
    # Load HTF data ONCE before loop for weekly calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Load daily data for volume and ATR filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (R3, S3) from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r3_1w = close_1w + (1.1 * (high_1w - low_1w) * 1.1 / 4.0)  # R3 = C + 1.1*(H-L)*1.1/4
    s3_1w = close_1w - (1.1 * (high_1w - low_1w) * 1.1 / 4.0)  # S3 = C - 1.1*(H-L)*1.1/4
    
    # Calculate daily ATR(14) and ATR(50) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original array
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr_14 > atr_50  # volatility expansion filter
    
    # Calculate daily volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Align weekly indicators to 6h timeframe (completed weekly bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Align daily filters to 6h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    volume_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for ATR(50) and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(volume_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_atr_ok = atr_aligned[i]
        curr_volume_ok = volume_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation, ATR filter, and breakout of weekly levels
            if curr_volume_ok and curr_atr_ok:
                # Bullish breakout: price breaks above weekly R3
                if curr_close > curr_r3:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below weekly S3
                elif curr_close < curr_s3:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to weekly pivot OR breaks below S3 with volume
            high_1w_i = df_1w['high'].values
            low_1w_i = df_1w['low'].values
            close_1w_i = df_1w['close'].values
            pivot_1w = (high_1w_i + low_1w_i + close_1w_i) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
            curr_pivot = pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else curr_r3  # fallback
            
            if curr_close <= curr_pivot or (curr_close < curr_s3 and curr_volume_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to weekly pivot OR breaks above R3 with volume
            high_1w_i = df_1w['high'].values
            low_1w_i = df_1w['low'].values
            close_1w_i = df_1w['close'].values
            pivot_1w = (high_1w_i + low_1w_i + close_1w_i) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
            curr_pivot = pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else curr_s3  # fallback
            
            if curr_close >= curr_pivot or (curr_close > curr_r3 and curr_volume_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals