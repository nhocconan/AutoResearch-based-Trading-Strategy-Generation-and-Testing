#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width regime + Camarilla pivot breakout with volume confirmation
# Long when BBW < 30th percentile (low volatility squeeze) AND price breaks above Camarilla R3 AND volume > 1.5 * 20-bar avg
# Short when BBW < 30th percentile AND price breaks below Camarilla S3 AND volume > 1.5 * 20-bar avg
# Exit when price re-enters the Camarilla R2-S2 range (mean reversion within the pivot zone)
# Uses discrete sizing 0.25 to control fee drag and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# BBW regime filter avoids whipsaws in high volatility, Camarilla provides structure, volume confirms participation
# Works in both bull and bear markets by fading extremes in low volatility regimes

name = "6h_BBW_Camarilla_R3S3_Breakout_Volume_v1"
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
    
    # Calculate Bollinger Band Width (20, 2) on 6h data
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid  # Normalized width
    
    # Calculate BBW percentile rank (30th percentile threshold for low volatility)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=100, min_periods=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    low_volatility = bb_width_percentile < 0.30  # BBW below 30th percentile
    
    # Get 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3/S3, R2/S2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    camarilla_r2 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_s2 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Calculate volume confirmation: volume > 1.5 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(low_volatility[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r2_aligned[i]) or 
            np.isnan(camarilla_s2_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Breakout entry with low volatility filter and volume confirmation
            # Long: Low volatility AND price breaks above R3 AND volume spike
            if low_volatility[i] and close[i] > camarilla_r3_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Low volatility AND price breaks below S3 AND volume spike
            elif low_volatility[i] and close[i] < camarilla_s3_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters R2-S2 range (mean reversion)
            if camarilla_s2_aligned[i] <= close[i] <= camarilla_r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters R2-S2 range (mean reversion)
            if camarilla_s2_aligned[i] <= close[i] <= camarilla_r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals