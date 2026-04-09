#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot + 12h volume regime + 1d trend filter
# - Primary signal: 6h price breaks above Camarilla R3 (long) or below S3 (short) from prior 12h
# - Volume confirmation: 12h volume > 50-period median volume (ensures participation)
# - Trend filter: 1d EMA200 - price must be above EMA200 for longs, below for shorts
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Camarilla levels provide adaptive support/resistance, volume confirms
#   breakout strength, 1d EMA200 ensures alignment with major trend reducing false signals

name = "6h_12h_1d_camarilla_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h indicators for Camarilla pivots (using prior completed 12h bar)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla levels from prior 12h bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low)
    #           S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    cam_r3_12h = close_12h + 1.125 * (high_12h - low_12h)
    cam_s3_12h = close_12h - 1.125 * (high_12h - low_12h)
    
    # Align Camarilla levels to 6h timeframe (completed 12h bar only)
    cam_r3_aligned = align_htf_to_ltf(prices, df_12h, cam_r3_12h)
    cam_s3_aligned = align_htf_to_ltf(prices, df_12h, cam_s3_12h)
    
    # 12h volume regime: volume > 50-period median volume
    median_volume_50 = pd.Series(volume_12h).rolling(window=50, min_periods=50).median().values
    volume_regime_12h = volume_12h > median_volume_50
    volume_regime_aligned = align_htf_to_ltf(prices, df_12h, volume_regime_12h)
    
    # 1d EMA200 for trend direction
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or
            np.isnan(volume_regime_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Camarilla S3 OR price crosses below EMA200
            if close[i] < cam_s3_aligned[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Camarilla R3 OR price crosses above EMA200
            if close[i] > cam_r3_aligned[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume confirmation and EMA200 filter
            # Long: price > Camarilla R3 AND volume regime AND price above EMA200
            if close[i] > cam_r3_aligned[i] and volume_regime_aligned[i] and close[i] > ema_200_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price < Camarilla S3 AND volume regime AND price below EMA200
            elif close[i] < cam_s3_aligned[i] and volume_regime_aligned[i] and close[i] < ema_200_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals