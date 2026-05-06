#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h volume confirmation and 1d EMA50 trend filter
# Long when BB width < 20th percentile (squeeze) AND price breaks above upper BB AND 12h volume > 1.5 * 20-bar avg AND 1d close > 1d EMA50
# Short when BB width < 20th percentile (squeeze) AND price breaks below lower BB AND 12h volume > 1.5 * 20-bar avg AND 1d close < 1d EMA50
# Exit when price retests the 20-period SMA (mean reversion within the squeeze context)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Bollinger squeeze identifies low volatility periods preceding breakouts
# Volume confirmation ensures participation, EMA50 filters for trend alignment
# Works in both bull and bear markets by following the 1d trend and trading breakouts from consolidation

name = "6h_BB_Squeeze_Breakout_12hVol_1dEMA50_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands and EMA50 ONCE before loop
    df_6h = get_htf_data(prices, '6h')  # Actually gets 6h data for BB
    df_12h = get_htf_data(prices, '12h')  # For volume confirmation
    df_1d = get_htf_data(prices, '1d')   # For EMA50 trend filter
    
    if len(df_6h) < 20 or len(df_12h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    volume_12h = df_12h['volume'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Bollinger Bands for 6h timeframe
    close_6h_series = pd.Series(close_6h)
    sma_20 = close_6h_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_6h_series.rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2.0 * std_20)
    bb_lower = sma_20 - (2.0 * std_20)
    bb_width = bb_upper - bb_lower
    
    # Calculate BB width percentile (20th) for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile_20 = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze_condition = bb_width < bb_width_percentile_20
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-bar average volume (12h timeframe)
    volume_12h_series = pd.Series(volume_12h)
    avg_volume_20 = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.5 * avg_volume_20)
    
    # Align HTF indicators to 6h timeframe (wait for completed bars)
    sma_20_aligned = align_htf_to_ltf(prices, df_6h, sma_20)
    bb_upper_aligned = align_htf_to_ltf(prices, df_6h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_6h, bb_lower)
    squeeze_aligned = align_htf_to_ltf(prices, df_6h, squeeze_condition.astype(float))
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(sma_20_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(squeeze_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        squeeze = bool(squeeze_aligned[i])
        vol_spike = bool(volume_spike_aligned[i])
        
        if position == 0:
            # Long breakout: squeeze AND price > upper BB AND uptrend AND volume spike
            if squeeze and close[i] > bb_upper_aligned[i] and close[i] > ema50_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short breakdown: squeeze AND price < lower BB AND downtrend AND volume spike
            elif squeeze and close[i] < bb_lower_aligned[i] and close[i] < ema50_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests 20-period SMA from above
            if close[i] <= sma_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests 20-period SMA from below
            if close[i] >= sma_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals