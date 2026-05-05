#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Bollinger Band squeeze breakout with daily EMA50 trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) AND daily EMA50 up AND volume > 1.5 * avg_volume(20)
# Short when price breaks below lower BB(20,2) AND daily EMA50 down AND volume > 1.5 * avg_volume(20)
# Exit when price crosses back through daily EMA50 OR Bollinger Band width expands above 50th percentile
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Bollinger Band squeeze identifies low volatility periods primed for breakout
# Daily EMA50 filters for primary trend alignment to avoid counter-trend trades
# Volume confirmation ensures breakout legitimacy
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "1d_BB_Squeeze_Breakout_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Bollinger Bands calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for BB
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands(20,2) on weekly timeframe
    close_1w_series = pd.Series(close_1w)
    bb_middle = close_1w_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1w_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Align Bollinger Bands to 1d timeframe (wait for completed weekly bar)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1w, bb_middle)
    
    # Calculate Bollinger Band width for regime filter (squeeze detection)
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_aligned = align_htf_to_ltf(prices, df_1w, bb_width)
    
    # Calculate 20-period percentile rank of BB width (squeeze = low percentile)
    bb_width_series = pd.Series(bb_width_aligned)
    bb_width_percentile = bb_width_series.rolling(window=100, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA50 slope for trend direction (up/down)
    ema50_slope = np.diff(ema50_1d_aligned, prepend=ema50_1d_aligned[0])
    ema50_up = ema50_slope > 0
    ema50_down = ema50_slope < 0
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper BB AND BB width percentile < 30 (squeeze) AND EMA50 up AND volume confirmation
            if (close[i] > bb_upper_aligned[i] and bb_width_percentile[i] < 0.3 and 
                ema50_up[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB AND BB width percentile < 30 (squeeze) AND EMA50 down AND volume confirmation
            elif (close[i] < bb_lower_aligned[i] and bb_width_percentile[i] < 0.3 and 
                  ema50_down[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below EMA50 OR BB width percentile > 70 (expansion)
            if close[i] < ema50_1d_aligned[i] or bb_width_percentile[i] > 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above EMA50 OR BB width percentile > 70 (expansion)
            if close[i] > ema50_1d_aligned[i] or bb_width_percentile[i] > 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals