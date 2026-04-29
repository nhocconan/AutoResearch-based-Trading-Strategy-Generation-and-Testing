#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with Weekly Trend Filter and Volume Spike
# Uses weekly Camarilla pivot levels (R3/S3) as strong support/resistance from 1w timeframe
# Breakouts above R3 or below S3 with volume confirmation capture strong momentum moves
# Weekly EMA50 filter ensures we only trade breakouts in the direction of the weekly trend
# Works in both bull and bear markets by aligning with higher timeframe trend
# Target: 15-25 trades/year (60-100 total over 4 years)

name = "4h_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for weekly calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (R3, S3) from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r3 = close_1w + (1.1 * (high_1w - low_1w) * 1.1 / 4.0)  # R3 = C + 1.1*(H-L)*1.1/4
    s3 = close_1w - (1.1 * (high_1w - low_1w) * 1.1 / 4.0)  # S3 = C - 1.1*(H-L)*1.1/4
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to 4h timeframe (completed 1w bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(55, 20)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema50 = ema_50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Determine weekly trend: price above/below EMA50
        uptrend = curr_close > curr_ema50
        downtrend = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in direction of weekly trend
            if curr_volume_confirm:
                # Bullish breakout: price breaks above R3 in uptrend
                if uptrend and curr_close > curr_r3:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below S3 in downtrend
                elif downtrend and curr_close < curr_s3:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to weekly pivot OR breaks below S3 with volume
            # Calculate weekly pivot for exit
            high_1w_i = df_1w['high'].values
            low_1w_i = df_1w['low'].values
            close_1w_i = df_1w['close'].values
            pivot_1w = (high_1w_i + low_1w_i + close_1w_i) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
            curr_pivot = pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else curr_r3  # fallback
            
            if curr_close <= curr_pivot or (curr_close < curr_s3 and curr_volume_confirm):
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
            
            if curr_close >= curr_pivot or (curr_close > curr_r3 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals