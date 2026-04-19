#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h momentum with weekly trend filter and volume confirmation
# Long when price > weekly VWAP, weekly trend up (close > weekly open), and volume spike
# Short when price < weekly VWAP, weekly trend down (close < weekly open), and volume spike
# Uses weekly VWAP as dynamic support/resistance and weekly trend as filter.
# Volume spike confirms institutional participation. Designed for 6h to capture multi-day moves.
# Target: 50-150 total trades over 4 years = 12-37/year.
name = "6h_WeeklyVWAP_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for VWAP and trend filter
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_open = df_weekly['open'].values
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_volume = df_weekly['volume'].values
    
    # Calculate weekly VWAP (volume weighted average price)
    vwap_weekly = (weekly_volume * (weekly_high + weekly_low + weekly_close) / 3).cumsum() / weekly_volume.cumsum()
    
    # Weekly trend: 1 if bullish (close > open), -1 if bearish (close < open), 0 otherwise
    weekly_trend = np.where(weekly_close > weekly_open, 1, np.where(weekly_close < weekly_open, -1, 0))
    
    # Align weekly VWAP and trend to 6h
    vwap_weekly_aligned = align_htf_to_ltf(prices, df_weekly, vwap_weekly)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend.astype(float))
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_weekly_aligned[i]) or np.isnan(weekly_trend_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_weekly_aligned[i]
        trend = weekly_trend_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Enter long if price above weekly VWAP, weekly trend up, and volume spike
            if price > vwap and trend > 0 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if price below weekly VWAP, weekly trend down, and volume spike
            elif price < vwap and trend < 0 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below weekly VWAP or trend turns bearish
            if price < vwap or trend < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above weekly VWAP or trend turns bullish
            if price > vwap or trend > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals