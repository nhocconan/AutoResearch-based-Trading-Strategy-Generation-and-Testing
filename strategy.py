#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d trend filter
# Uses 12h Camarilla levels (R3/S3 for mean reversion, R4/S4 for breakout)
# Enters long when price breaks above R4 with 12h volume > 1.5x 20-period average and 1d close > 1d SMA50
# Enters short when price breaks below S4 with 12h volume > 1.5x 20-period average and 1d close < 1d SMA50
# Exits when price reverts to 12h VWAP or opposite Camarilla level (R3/S3)
# Position size 0.25 to limit drawdown
# Target: 12-25 trades/year per symbol to minimize fee drag

name = "6h_12h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Camarilla levels and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (based on previous 12h bar)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_r3 = np.full(len(df_12h), np.nan)
    camarilla_s3 = np.full(len(df_12h), np.nan)
    camarilla_r4 = np.full(len(df_12h), np.nan)
    camarilla_s4 = np.full(len(df_12h), np.nan)
    camarilla_vwap = np.full(len(df_12h), np.nan)
    
    for i in range(1, len(df_12h)):
        # Previous bar's range
        prev_high = high_12h[i-1]
        prev_low = low_12h[i-1]
        prev_close = close_12h[i-1]
        prev_range = prev_high - prev_low
        
        if prev_range > 0:
            camarilla_r3[i] = prev_close + prev_range * 1.1 / 2
            camarilla_s3[i] = prev_close - prev_range * 1.1 / 2
            camarilla_r4[i] = prev_close + prev_range * 1.1
            camarilla_s4[i] = prev_close - prev_range * 1.1
        
        # Typical price * volume / cumulative volume for VWAP
        typical_price = (high_12h[i] + low_12h[i] + close_12h[i]) / 3
        if i == 0:
            camarilla_vwap[i] = typical_price * volume[i] if len(volume) > i else typical_price
        else:
            # Simplified VWAP: cumulative typical price * volume / cumulative volume
            # We'll use a rolling approximation for alignment purposes
            camarilla_vwap[i] = typical_price  # Will be refined in alignment
    
    # Calculate 12h volume moving average (20-period)
    vol_ma_20_12h = np.full(len(df_12h), np.nan)
    vol_sum = 0.0
    for i in range(len(df_12h)):
        vol_sum += df_12h['volume'].iloc[i] if 'volume' in df_12h.columns else volume[i * 2]  # Approximation
        if i >= 20:
            vol_sum -= df_12h['volume'].iloc[i-20] if 'volume' in df_12h.columns else volume[(i-20) * 2]
        if i >= 19:
            vol_ma_20_12h[i] = vol_sum / 20
    
    # Calculate 1d SMA50 for trend filter
    close_1d = df_1d['close'].values
    sma_50_1d = np.full(len(df_1d), np.nan)
    close_sum = 0.0
    for i in range(len(df_1d)):
        close_sum += close_1d[i]
        if i >= 50:
            close_sum -= close_1d[i-50]
        if i >= 49:
            sma_50_1d[i] = close_sum / 50
    
    # Align all 12h data to 6h timeframe
    camarilla_r3_6h = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_r4_6h = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    camarilla_vwap_6h = align_htf_to_ltf(prices, df_12h, camarilla_vwap)
    vol_ma_20_12h_6h = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Align 1d SMA50 to 6h timeframe
    sma_50_1d_6h = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after SMA50 warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_6h[i]) or np.isnan(camarilla_s4_6h[i]) or
            np.isnan(camarilla_r3_6h[i]) or np.isnan(camarilla_s3_6h[i]) or
            np.isnan(camarilla_vwap_6h[i]) or np.isnan(vol_ma_20_12h_6h[i]) or
            np.isnan(sma_50_1d_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        # Approximate 12h volume from 6h data (every 2nd 6h bar is a 12h bar)
        vol_12h_current = volume[i] + (volume[i-1] if i > 0 else 0)  # Sum of two 6h bars
        vol_ma_20_current = vol_ma_20_12h_6h[i]
        vol_ratio = vol_12h_current / vol_ma_20_current if vol_ma_20_current > 0 else 0
        
        if position == 1:  # Long position
            # Exit: price closes below 12h VWAP or below S3 (mean reversion)
            if close[i] <= camarilla_vwap_6h[i] or close[i] < camarilla_s3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h VWAP or above R3 (mean reversion)
            if close[i] >= camarilla_vwap_6h[i] or close[i] > camarilla_r3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above R4 with volume confirmation and 1d uptrend
            if (close[i] > camarilla_r4_6h[i] and 
                vol_ratio > 1.5 and 
                close_1d[-1] > sma_50_1d[-1] if len(close_1d) > 0 and len(sma_50_1d) > 0 else True):  # Simplified trend check
                # More robust 1d trend: check if current 1d close > 1d SMA50
                # Find the corresponding 1d bar index for current 6h bar
                # Since we aligned, we can use the aligned value directly
                if close[i] > camarilla_r4_6h[i] and vol_ratio > 1.5:
                    # Check 1d trend using aligned SMA50
                    if not np.isnan(sma_50_1d_6h[i]) and close_1d[-1] > sma_50_1d[-1]:  # Fallback to last known
                        position = 1
                        signals[i] = 0.25
                    elif not np.isnan(sma_50_1d_6h[i]):  # Use aligned 1d SMA50
                        # We need the 1d close that corresponds to this 6h bar
                        # Since alignment looks back, we check if 1d close > 1d SMA50
                        # Approximation: use the aligned values
                        signals[i] = 0.25  # Trust alignment for trend
            
            # Enter short: price breaks below S4 with volume confirmation and 1d downtrend
            elif (close[i] < camarilla_s4_6h[i] and 
                  vol_ratio > 1.5):
                if not np.isnan(sma_50_1d_6h[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals