#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot breakout with daily volume spike and chop regime filter
# Long when price closes above H3 (Camarilla resistance) AND volume > 1.8x 20-period average AND chop > 61.8 (range market)
# Short when price closes below L3 (Camarilla support) AND volume > 1.8x 20-period average AND chop > 61.8
# Exit when price crosses back to the pivot level (median of H3/L3)
# Uses Camarilla levels for intraday support/resistance, volume for breakout confirmation, chop to avoid trending markets
# Target: 50-120 total trades over 4 years (12-30/year) to minimize fee drag while capturing mean reversion in ranging markets

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate pivot points from previous day (using daily OHLC)
    # We need daily data to calculate Camarilla levels for current 4h bar
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # H2 = close + 1.0 * (high - low)
    # H1 = close + 0.75 * (high - low)
    # L1 = close - 0.75 * (high - low)
    # L2 = close - 1.0 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # L4 = close - 1.5 * (high - low)
    
    # Calculate daily ranges
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels (we'll use H3 and L3 for entries)
    camarilla_h3 = daily_close + 1.25 * (daily_high - daily_low)
    camarilla_l3 = daily_close - 1.25 * (daily_high - daily_low)
    camarilla_pivot = (daily_high + daily_low + daily_close) / 3  # Classic pivot
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate Choppiness Index (using daily data for regime detection)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    # We'll use a simplified version: high-low range over period
    period = 14
    high_low_range = pd.Series(daily_high - daily_low).rolling(window=period, min_periods=period).sum().values
    max_high = pd.Series(daily_high).rolling(window=period, min_periods=period).max().values
    min_low = pd.Series(daily_low).rolling(window=period, min_periods=period).min().values
    chop = 100 * np.log10(high_low_range / (max_high - min_low + 1e-10)) / np.log10(period)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate volume average for confirmation (20-period on 4h)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 30)  # Need 20 for vol avg, 30 for chop calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.8
        
        if position == 0:
            # Long setup: price closes above H3 + volume confirmation + chop > 61.8 (range market)
            if (price > camarilla_h3_aligned[i] and vol > vol_threshold and chop_aligned[i] > 61.8):
                position = 1
                signals[i] = position_size
            # Short setup: price closes below L3 + volume confirmation + chop > 61.8
            elif (price < camarilla_l3_aligned[i] and vol > vol_threshold and chop_aligned[i] > 61.8):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back to pivot level (or below)
            if price < camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back to pivot level (or above)
            if price > camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Volume_Chop"
timeframe = "4h"
leverage = 1.0