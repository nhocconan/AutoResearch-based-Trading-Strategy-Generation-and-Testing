#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Reversal with 1d Trend Filter and Volume Confirmation
# Uses daily Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
# filtered by 1d ADX > 25 for trend strength and volume > 1.5x 20-period average.
# In trending markets (ADX>25): breakout continuation at R4/S4 levels.
# In ranging markets (ADX<25): mean reversion at R3/S3 levels.
# Designed to work in both bull and bear markets by adapting to regime.
# Target: 15-35 trades/year via Camarilla + trend + volume confluence.

name = "6h_camarilla_1d_trend_volume_v1"
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
    
    # Get 1d data for Camarilla pivots and ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX for trend strength on 1d data
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # where C, H, L are from previous day
    cam_close = np.concatenate([[np.nan], close_1d[:-1]])  # previous day close
    cam_high = np.concatenate([[np.nan], high_1d[:-1]])    # previous day high
    cam_low = np.concatenate([[np.nan], low_1d[:-1]])      # previous day low
    
    cam_range = cam_high - cam_low
    r4 = cam_close + (cam_range * 1.1 / 2)
    r3 = cam_close + (cam_range * 1.1 / 4)
    s3 = cam_close - (cam_range * 1.1 / 4)
    s4 = cam_close - (cam_range * 1.1 / 2)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d values for current 6h bar
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)[i]
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)[i]
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)[i]
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)[i]
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)[i]
        
        # Regime filters
        trending = adx_aligned > 25      # Strong trend
        ranging = adx_aligned < 25       # Ranging/weak trend
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if trending:
                # In trend: exit on bearish break below S4
                if close[i] < s4_aligned:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                # In range: exit on reversion to mean (pivot) or opposite S3 break
                if close[i] <= cam_close[i] or close[i] < s3_aligned:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if trending:
                # In trend: exit on bullish break above R4
                if close[i] > r4_aligned:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                # In range: exit on reversion to mean (pivot) or opposite R3 break
                if close[i] >= cam_close[i] or close[i] > r3_aligned:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_confirm:
                if trending:
                    # Trend following: breakout continuation
                    if close[i] > r4_aligned:  # Bullish breakout
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < s4_aligned:  # Bearish breakout
                        position = -1
                        signals[i] = -0.25
                else:
                    # Mean reversion: fade at extreme levels
                    if close[i] <= s3_aligned:  # Oversold
                        position = 1
                        signals[i] = 0.25
                    elif close[i] >= r3_aligned:  # Overbought
                        position = -1
                        signals[i] = -0.25
    
    return signals