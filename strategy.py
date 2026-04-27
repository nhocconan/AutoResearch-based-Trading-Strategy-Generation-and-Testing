#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Bollinger Band width contraction (squeeze) followed by breakout.
# Bollinger Band width measures volatility contraction (squeeze). Breakout from squeeze with volume
# indicates strong directional move. Uses weekly trend filter (price > weekly EMA50) to avoid
# counter-trend trades. Designed for low trade frequency (<20/year) to minimize fee drag.
# Works in bull markets (bullish breakouts from squeeze) and bear markets (bearish breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Bollinger Bands (20, 2.0)
    bb_length = 20
    bb_mult = 2.0
    
    # Basis (SMA)
    basis_1w = pd.Series(close_1w).rolling(window=bb_length, min_periods=bb_length).mean().values
    # Deviation
    dev_1w = bb_mult * pd.Series(close_1w).rolling(window=bb_length, min_periods=bb_length).std().values
    # Upper and Lower bands
    upper_1w = basis_1w + dev_1w
    lower_1w = basis_1w - dev_1w
    # Bandwidth
    bw_1w = (upper_1w - lower_1w) / basis_1w * 100  # Percent
    
    # Bollinger Band width squeeze: bandwidth < 20th percentile of last 50 weeks
    bw_percentile = pd.Series(bw_1w).rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze = bw_1w < bw_percentile
    
    # Weekly trend filter: price > EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all weekly indicators to daily
    basis_1w_aligned = align_htf_to_ltf(prices, df_1w, basis_1w)
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze.astype(float))
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(basis_1w_aligned[i]) or np.isnan(upper_1w_aligned[i]) or 
            np.isnan(lower_1w_aligned[i]) or np.isnan(squeeze_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above upper BB with squeeze, uptrend, and volume
        if (close[i] > upper_1w_aligned[i] and 
            squeeze_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short entry: price breaks below lower BB with squeeze, downtrend, and volume
        elif (close[i] < lower_1w_aligned[i] and 
              squeeze_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to BB middle (mean reversion within squeeze context)
        elif position == 1 and close[i] <= basis_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] >= basis_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_BB_Width_Squeeze_Breakout_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0