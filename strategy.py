#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h weekly Bollinger Band squeeze breakout with volume confirmation
# Long when price breaks above upper BB with low volatility (BBW < 20th percentile) and volume > 1.5x 12h average
# Short when price breaks below lower BB with low volatility (BBW < 20th percentile) and volume > 1.5x 12h average
# Exit when price crosses the 12-period SMA
# Uses weekly BB squeeze to identify low volatility periods before breakouts in both bull and bear markets
# Target: 12-37 trades/year per symbol to stay within frequency limits
name = "12h_Weekly_BB_Squeeze_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(df_1w['close']).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(df_1w['close']).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std * std_20)
    lower_band = sma_20 - (bb_std * std_20)
    bb_width = ((upper_band - lower_band) / sma_20) * 100  # Percentage
    
    # Calculate 20th percentile of BB width for squeeze condition
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile_20 = bb_width_series.rolling(window=50, min_periods=10).quantile(0.20).values
    
    # Align weekly data to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    sma_aligned = align_htf_to_ltf(prices, df_1w, sma_20)
    bb_width_aligned = align_htf_to_ltf(prices, df_1w, bb_width)
    bb_percentile_aligned = align_htf_to_ltf(prices, df_1w, bb_width_percentile_20)
    
    # Get 12h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(sma_aligned[i]) or np.isnan(bb_width_aligned[i]) or 
            np.isnan(bb_percentile_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        sma = sma_aligned[i]
        bb_width = bb_width_aligned[i]
        bb_percentile = bb_percentile_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Squeeze condition: low volatility (BB width below 20th percentile)
        squeeze_condition = bb_width < bb_percentile
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above upper BB with squeeze and volume confirmation
            if i > 0 and close[i-1] <= upper_aligned[i-1] and price > upper and squeeze_condition and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower BB with squeeze and volume confirmation
            elif i > 0 and close[i-1] >= lower_aligned[i-1] and price < lower and squeeze_condition and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 12-period SMA
            if price < sma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 12-period SMA
            if price > sma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals