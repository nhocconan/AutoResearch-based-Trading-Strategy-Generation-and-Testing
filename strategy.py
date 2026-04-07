#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla pivot + 1d trend filter + volume confirmation
# Hypothesis: Camarilla levels (R3/S3, R4/S4) act as institutional support/resistance.
# In trending markets (1d EMA25), breakouts beyond R4/S4 continue; reversions at R3/S3.
# Works in both bull/bear: trend filter adapts direction, volume avoids false breakouts.
# Target: 15-35 trades/year (~60-140 total over 4 years) to minimize fee drag.
name = "6h_camarilla_1d_trend_volume_v2"
timeframe = "6h"
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
    
    # Get daily data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day
    # Classic formula: based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R3 = prev_close + range_val * 1.1 / 2
    R4 = prev_close + range_val * 1.1
    S3 = prev_close - range_val * 1.1 / 2
    S4 = prev_close - range_val * 1.1
    
    # Align to 6h timeframe (shifted by 1 day for lookback)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Daily trend filter: EMA25
    ema_25 = df_1d['close'].ewm(span=25, min_periods=25).mean().values
    ema_25_6h = align_htf_to_ltf(prices, df_1d, ema_25)
    
    # Volume confirmation: 60-period volume MA on 6h
    vol_ma_60 = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(R3_6h[i]) or np.isnan(R4_6h[i]) or 
            np.isnan(S3_6h[i]) or np.isnan(S4_6h[i]) or
            np.isnan(ema_25_6h[i]) or np.isnan(vol_ma_60[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 60-period average
        vol_confirm = volume[i] > vol_ma_60[i]
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (reversion to mean)
            if close[i] < S3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above R3 (reversion to mean)
            if close[i] > R3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Determine trend direction from daily EMA25
            uptrend = close[i] > ema_25_6h[i]
            
            # Enter long: break above R4 in uptrend OR reversal at S3 in downtrend
            if vol_confirm:
                if uptrend and close[i] > R4_6h[i]:
                    position = 1
                    signals[i] = 0.25
                elif (not uptrend) and close[i] < S3_6h[i]:
                    position = 1
                    signals[i] = 0.25
            # Enter short: break below S4 in downtrend OR reversal at R3 in uptrend
            if vol_confirm:
                if (not uptrend) and close[i] < S4_6h[i]:
                    position = -1
                    signals[i] = -0.25
                elif uptrend and close[i] > R3_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals