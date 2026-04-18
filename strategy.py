#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Bollinger Band breakout with daily volume confirmation
# Works in bull markets (breakouts continue) and bear markets (mean reversion at bands)
# Uses weekly timeframe for structure, daily for entry - low frequency to avoid fee drag
# Target: 15-25 trades/year, fits 1d timeframe constraints

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate weekly Bollinger Bands (20, 2.0)
    weekly_close_series = pd.Series(weekly_close)
    bb_middle = weekly_close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = weekly_close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Align weekly BB to daily timeframe (only after weekly bar closes)
    bb_middle_daily = align_htf_to_ltf(prices, df_weekly, bb_middle)
    bb_upper_daily = align_htf_to_ltf(prices, df_weekly, bb_upper)
    bb_lower_daily = align_htf_to_ltf(prices, df_weekly, bb_lower)
    
    # Daily volume confirmation (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # need BB and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_middle_daily[i]) or np.isnan(bb_upper_daily[i]) or 
            np.isnan(bb_lower_daily[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above upper BB with volume
            if close[i] > bb_upper_daily[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower BB with volume
            elif close[i] < bb_lower_daily[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to middle BB (mean reversion)
            if close[i] < bb_middle_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle BB (mean reversion)
            if close[i] > bb_middle_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "Weekly_Bollinger_Breakout_Daily_Volume"
timeframe = "1d"
leverage = 1.0