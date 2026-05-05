#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily Bollinger Band squeeze breakout with 6h volume confirmation and 12h EMA50 trend filter
# Long when price breaks above upper BB(20,2) AND BB width < 20th percentile (squeeze) AND volume > 1.5 * avg_volume(20) AND price > 12h EMA50
# Short when price breaks below lower BB(20,2) AND BB width < 20th percentile (squeeze) AND volume > 1.5 * avg_volume(20) AND price < 12h EMA50
# Exit when price returns to BB middle (mean reversion) OR BB width expands above 50th percentile (squeeze end)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Bollinger squeeze identifies low volatility primed for breakout
# Volume confirmation ensures breakout validity
# 12h EMA50 filters for primary trend alignment to avoid counter-trend trades
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "6h_BollingerSqueeze_Breakout_12hEMA50_Volume"
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
    
    # Get 1d data ONCE before loop for Bollinger Bands calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 daily bars for BB
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands(20,2) on daily timeframe
    close_1d_series = pd.Series(close_1d)
    bb_middle = close_1d_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1d_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # Normalized width
    
    # Align Bollinger Bands to 6h timeframe (wait for completed daily bar)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(bb_middle_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(bb_width_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper BB AND BB width < 20th percentile (squeeze) AND volume confirmation AND above 12h EMA50
            if (close[i] > bb_upper_aligned[i] and 
                bb_width_aligned[i] < np.percentile(bb_width_aligned[:i+1], 20) and 
                volume_confirm[i] and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB AND BB width < 20th percentile (squeeze) AND volume confirmation AND below 12h EMA50
            elif (close[i] < bb_lower_aligned[i] and 
                  bb_width_aligned[i] < np.percentile(bb_width_aligned[:i+1], 20) and 
                  volume_confirm[i] and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to BB middle OR BB width expands above 50th percentile (squeeze end)
            if (close[i] <= bb_middle_aligned[i] or 
                bb_width_aligned[i] > np.percentile(bb_width_aligned[:i+1], 50)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to BB middle OR BB width expands above 50th percentile (squeeze end)
            if (close[i] >= bb_middle_aligned[i] or 
                bb_width_aligned[i] > np.percentile(bb_width_aligned[:i+1], 50)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals