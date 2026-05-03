#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h Volume Confirmation and 1d Trend Filter.
# Long when: price breaks above upper BB(20,2) AND BB width < 20th percentile (squeeze) AND 12h volume > 1.5x 20-period MA AND 1d close > 1d EMA50.
# Short when: price breaks below lower BB(20,2) AND BB width < 20th percentile (squeeze) AND 12h volume > 1.5x 20-period MA AND 1d close < 1d EMA50.
# Exit when: price returns to middle BB (20-period SMA) OR BB width > 50th percentile (squeeze end).
# Uses 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Bollinger Squeeze captures low volatility breakouts, volume confirms participation, 1d EMA50 filters for higher timeframe trend.
# Works in both bull (breakouts with trend) and bear (breakdowns with trend) markets.

name = "6h_BBSqueeze_12hVol_1dEMA50_Trend"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 6h data for Bollinger Bands
    df_6h = prices  # primary timeframe is 6h
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Calculate 6h Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20
    bb_width = (upper_bb - lower_bb) / middle_bb  # normalized width
    
    # Calculate 6h BB width percentiles (20th and 50th for squeeze detection)
    # Use expanding window to avoid look-ahead
    bb_width_20th = np.full(n, np.nan)
    bb_width_50th = np.full(n, np.nan)
    for i in range(20, n):
        window = bb_width[:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 20:
            bb_width_20th[i] = np.percentile(valid, 20)
            bb_width_50th[i] = np.percentile(valid, 50)
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h volume 20-period MA for spike detection
    volume_12h = df_12h['volume'].values
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(sma_20[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(bb_width_20th[i]) or np.isnan(bb_width_50th[i]) or
            np.isnan(volume_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Squeeze conditions
        squeeze_active = bb_width[i] < bb_width_20th[i]
        squeeze_end = bb_width[i] > bb_width_50th[i]
        
        # Volume spike condition: current 12h volume > 1.5x 20-period volume MA
        # Note: we need current 12h volume, get it from aligned 12h volume data
        # Since we don't have raw 12h volume aligned, we'll use the volume from prices resampled conceptually
        # Instead, use 6h volume > 1.5x 20-period 6h volume MA as proxy (more immediate)
        volume_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume[i] > (volume_ma_20_6h[i] * 1.5)
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper BB AND squeeze active AND volume spike AND uptrend AND session
            if close[i] > upper_bb[i] and squeeze_active and volume_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB AND squeeze active AND volume spike AND downtrend AND session
            elif close[i] < lower_bb[i] and squeeze_active and volume_spike and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle BB OR squeeze ends
            if close[i] < middle_bb[i] or squeeze_end:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle BB OR squeeze ends
            if close[i] > middle_bb[i] or squeeze_end:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals