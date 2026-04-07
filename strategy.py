#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams %R with weekly trend filter and volume confirmation
# Williams %R(14) identifies overbought/oversold conditions
# Weekly trend filter ensures we trade with the higher timeframe trend
# Volume confirmation reduces false signals
# Target: 12-37 trades/year, works in both bull and bear markets via mean reversion in trends
name = "6h_williamsr_weekly_trend_volume_v1"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend direction
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Calculate Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(weekly_ema_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Weekly trend filter: price above/below weekly EMA
        price_above_weekly = close[i] > weekly_ema_aligned[i]
        price_below_weekly = close[i] < weekly_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR trend changes
            if williams_r[i] >= -20 or not price_above_weekly:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR trend changes
            if williams_r[i] <= -80 or not price_below_weekly:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Williams %R crosses below -80 (oversold) + volume confirmation + weekly uptrend
            if williams_r[i] < -80 and vol_confirm and price_above_weekly:
                position = 1
                signals[i] = 0.25
            # Enter short: Williams %R crosses above -20 (overbought) + volume confirmation + weekly downtrend
            elif williams_r[i] > -20 and vol_confirm and price_below_weekly:
                position = -1
                signals[i] = -0.25
    
    return signals