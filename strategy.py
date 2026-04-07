#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Camarilla Pivot Reversal with Volume Confirmation and 1d Trend Filter
# Hypothesis: Price reversals at Camarilla pivot levels (S3/S4 for long, R3/R4 for short) 
# with volume spikes and aligned daily trend work across bull/bear markets.
# Uses 1d EMA for trend filter and Camarilla levels from prior day for entries.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "12h_camarilla_pivot_reversal_1d_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: 
    #   S1 = C - (H-L)*1.1/12
    #   S2 = C - (H-L)*1.1/6
    #   S3 = C - (H-L)*1.1/4
    #   S4 = C - (H-L)*1.1/2
    #   R4 = C + (H-L)*1.1/2
    #   R3 = C + (H-L)*1.1/4
    #   R2 = C + (H-L)*1.1/6
    #   R1 = C + (H-L)*1.1/12
    camarilla_S4 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    camarilla_R3 = np.full(n, np.nan)
    camarilla_R4 = np.full(n, np.nan)
    
    for i in range(1, len(df_1d)):
        # Use previous day's OHLC
        prev_high = df_1d['high'].iloc[i-1]
        prev_low = df_1d['low'].iloc[i-1]
        prev_close = df_1d['close'].iloc[i-1]
        
        range_hl = prev_high - prev_low
        if range_hl <= 0:
            continue
            
        # Calculate levels
        s4 = prev_close - (range_hl * 1.1 / 2)
        s3 = prev_close - (range_hl * 1.1 / 4)
        r3 = prev_close + (range_hl * 1.1 / 4)
        r4 = prev_close + (range_hl * 1.1 / 2)
        
        # Map to 12h timeframe: each 1d bar = 2 bars of 12h
        start_idx = i * 2
        end_idx = start_idx + 2
        if end_idx <= n:
            camarilla_S4[start_idx:end_idx] = s4
            camarilla_S3[start_idx:end_idx] = s3
            camarilla_R3[start_idx:end_idx] = r3
            camarilla_R4[start_idx:end_idx] = r4
    
    # Calculate volume average (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_S3[i]) or np.isnan(camarilla_S4[i]) or
            np.isnan(camarilla_R3[i]) or np.isnan(camarilla_R4[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition (at least 1.5x average)
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price reaches S3 (take profit) or trend changes to down
            if close[i] >= camarilla_S3[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price reaches R3 (take profit) or trend changes to up
            if close[i] <= camarilla_R3[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price at S4 with volume spike in uptrend
            if (close[i] <= camarilla_S4[i] and vol_spike and 
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price at R4 with volume spike in downtrend
            elif (close[i] >= camarilla_R4[i] and vol_spike and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals