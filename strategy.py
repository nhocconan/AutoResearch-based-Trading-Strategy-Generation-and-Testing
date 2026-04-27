#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Bollinger Band mean reversion on 1d timeframe with volume confirmation.
# In ranging markets (common in 2025-2026 BTC/ETH), price tends to revert to the mean after touching
# weekly Bollinger Bands. Volume spike confirms the reversal. Works in both bull and bear markets
# as it's a mean-reversion strategy, not trend-following. Target: 50-80 trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands (20-week, 2 std)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate Bollinger Bands on weekly data
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2.0 * std_20
    lower_band = sma_20 - 2.0 * std_20
    
    # Align weekly Bollinger Bands to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    sma_20_aligned = align_htf_to_ltf(prices, df_1w, sma_20)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-week BB + 20-day volume MA
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(sma_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        middle = sma_20_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price touches or goes below lower band + volume spike → expect reversion to mean
            if price <= lower and vol_filter:
                signals[i] = size
                position = 1
            # Short: price touches or goes above upper band + volume spike → expect reversion to mean
            elif price >= upper and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle (SMA) or touches upper band
            if price >= middle or price >= upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle (SMA) or touches lower band
            if price <= middle or price <= lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Weekly_Bollinger_MeanReversion_Volume"
timeframe = "1d"
leverage = 1.0