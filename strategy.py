#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Primary signal: 6h price breaks above Camarilla R4 or below S4 (strong breakout)
# - Trend filter: 1d EMA200 - only take longs when price > EMA200, shorts when price < EMA200
# - Volume confirmation: 6h volume > 1.5x 20-period median volume (ensure participation)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Camarilla R4/S4 breaks capture strong momentum, EMA200 filter ensures
#   alignment with higher timeframe trend, reducing false signals in choppy markets

name = "6h_1d_camarilla_breakout_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:  # Need enough for EMA200
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA200 for trend direction
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 6h timeframe (completed 1d bar only)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Camarilla pivot levels (based on previous day's OHLC)
    # Calculate daily pivot from 1d data, then derive Camarilla levels
    # Camarilla: R4 = Close + 1.5 * (High - Low), S4 = Close - 1.5 * (High - Low)
    camarilla_r4_1d = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_s4_1d = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    # 6h volume regime: volume > 1.5x 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Camarilla R3 OR below EMA200
            camarilla_r3_1d = close_1d + 1.125 * (high_1d - low_1d)
            camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
            if (i < len(camarilla_r3_aligned) and 
                not np.isnan(camarilla_r3_aligned[i]) and
                (close[i] < camarilla_r3_aligned[i] or close[i] < ema_200_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Camarilla S3 OR above EMA200
            camarilla_s3_1d = close_1d - 1.125 * (high_1d - low_1d)
            camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
            if (i < len(camarilla_s3_aligned) and 
                not np.isnan(camarilla_s3_aligned[i]) and
                (close[i] > camarilla_s3_aligned[i] or close[i] > ema_200_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla R4/S4 breakouts with volume confirmation and EMA200 filter
            # Long: price breaks above Camarilla R4 AND volume regime AND price > EMA200
            if (close[i] > camarilla_r4_aligned[i] and 
                volume_regime[i] and 
                close[i] > ema_200_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Camarilla S4 AND volume regime AND price < EMA200
            elif (close[i] < camarilla_s4_aligned[i] and 
                  volume_regime[i] and 
                  close[i] < ema_200_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals