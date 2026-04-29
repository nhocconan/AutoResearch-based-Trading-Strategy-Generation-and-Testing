#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 Breakout with Daily Trend Filter and Volume Spike
# Uses daily Camarilla pivot levels (R3/S3) as strong support/resistance from 1d timeframe
# Breakouts above R3 or below S3 with volume confirmation capture strong momentum moves
# Daily EMA34 filter ensures we only trade breakouts in the direction of the daily trend
# Works in both bull and bear markets by aligning with higher timeframe trend
# Target: 12-25 trades/year (50-100 total over 4 years)

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R3, S3) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r3 = close_1d + (1.1 * (high_1d - low_1d) * 1.1 / 4.0)  # R3 = C + 1.1*(H-L)*1.1/4
    s3 = close_1d - (1.1 * (high_1d - low_1d) * 1.1 / 4.0)  # S3 = C - 1.1*(H-L)*1.1/4
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily indicators to 6h timeframe (completed 1d bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_34_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema34 = ema_34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Determine daily trend: price above/below EMA34
        uptrend = curr_close > curr_ema34
        downtrend = curr_close < curr_ema34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in direction of daily trend
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
            # Exit when: price returns to daily pivot OR breaks below S3 with volume
            # Calculate daily pivot for exit
            high_1d_i = df_1d['high'].values
            low_1d_i = df_1d['low'].values
            close_1d_i = df_1d['close'].values
            pivot_1d = (high_1d_i + low_1d_i + close_1d_i) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
            curr_pivot = pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else curr_r3  # fallback
            
            if curr_close <= curr_pivot or (curr_close < curr_s3 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to daily pivot OR breaks above R3 with volume
            high_1d_i = df_1d['high'].values
            low_1d_i = df_1d['low'].values
            close_1d_i = df_1d['close'].values
            pivot_1d = (high_1d_i + low_1d_i + close_1d_i) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
            curr_pivot = pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else curr_s3  # fallback
            
            if curr_close >= curr_pivot or (curr_close > curr_r3 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals