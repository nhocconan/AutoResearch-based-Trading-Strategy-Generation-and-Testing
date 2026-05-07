#!/usr/bin/env python3
"""
1D_Bollinger_Bandwidth_Breakout_WeeklyTrend_Volume
Hypothesis: Daily breakouts above/below Bollinger Bands with weekly trend confirmation and volume spikes capture strong moves while avoiding whipsaws. Bollinger Bandwidth filter identifies low-volatility periods (squeeze) where breakouts are more meaningful. Works in bull/bear markets by following weekly trend direction. Targets 15-25 trades/year to minimize fee drag on daily timeframe.
"""
name = "1D_Bollinger_Bandwidth_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend direction
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend direction
    close_weekly_series = pd.Series(df_weekly['close'])
    weekly_ema50 = close_weekly_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Calculate daily Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean().values
    std20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + (2 * std20)
    lower_band = sma20 - (2 * std20)
    
    # Calculate Bollinger Bandwidth (normalized bandwidth for regime filter)
    bb_width = (upper_band - lower_band) / sma20
    # Bollinger Bandwidth percentile over 50 periods to identify low volatility (squeeze)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(weekly_ema50_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 20 days between trades to reduce frequency
            if bars_since_exit < 20:
                continue
                
            # Volume filter: current daily volume > 1.8 x 20-day average volume
            vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            volume_filter = volume[i] > (vol_avg[i] * 1.8) if not np.isnan(vol_avg[i]) else False
            
            # Long: price breaks above upper Bollinger Band with weekly uptrend, low volatility squeeze, and volume spike
            if (close[i] > upper_band[i] and close[i-1] <= upper_band[i-1] and 
                close[i] > weekly_ema50_aligned[i] and bb_width_percentile[i] < 0.3 and volume_filter):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below lower Bollinger Band with weekly downtrend, low volatility squeeze, and volume spike
            elif (close[i] < lower_band[i] and close[i-1] >= lower_band[i-1] and 
                  close[i] < weekly_ema50_aligned[i] and bb_width_percentile[i] < 0.3 and volume_filter):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to the opposite Bollinger Band (mean reversion within the band)
            if position == 1 and close[i] < sma20[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > sma20[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals