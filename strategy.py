#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with 1-day ATR-based volatility filter and volume confirmation.
# Long when: BB width at 20-day low + price breaks above upper BB(20,2) + volume > 1.5x 20-period average
# Short when: BB width at 20-day low + price breaks below lower BB(20,2) + volume > 1.5x 20-period average
# Exit when price crosses back inside Bollinger Bands (mean reversion of squeeze breakout).
# Bollinger squeeze captures low volatility periods before explosive moves, effective in both bull and bear markets.
# Target: 15-25 trades/year per symbol. Uses Bollinger Bands with standard deviation for volatility measurement.
name = "12h_BollingerSqueeze_Volume_Breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Bollinger Bands and volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands on daily data (20,2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma + bb_std * std
    lower_bb = sma - bb_std * std
    bb_width = upper_bb - lower_bb
    
    # BB width percentile rank (252-day lookback for 1-year context)
    bb_width_rank = np.full_like(bb_width, np.nan)
    lookback = 252
    for i in range(lookback, len(bb_width)):
        window = bb_width[i-lookback:i]
        if not np.all(np.isnan(window)):
            bb_width_rank[i] = np.percentile(window, 100 * (bb_width[i] - np.nanmin(window)) / (np.nanmax(window) - np.nanmin(window) + 1e-10))
    
    # Identify squeeze: BB width at or below 10th percentile (low volatility)
    squeeze = bb_width_rank <= 10
    
    # Align squeeze signal and Bollinger Bands to 12h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period + lookback, 20)  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(squeeze_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        is_squeeze = squeeze_aligned[i]
        upper_band = upper_bb_aligned[i]
        lower_band = lower_bb_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: squeeze condition + price breaks above upper BB + volume confirmation
            if is_squeeze and price > upper_band and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: squeeze condition + price breaks below lower BB + volume confirmation
            elif is_squeeze and price < lower_band and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back inside Bollinger Bands (mean reversion)
            if price < upper_band and price > lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back inside Bollinger Bands (mean reversion)
            if price < upper_band and price > lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals