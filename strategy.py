#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Mean Reversion at Daily VWAP Bands with 1d Volume Spike and 1w Trend Filter
# Trades when price deviates significantly from daily VWAP (mean reversion) with volume confirmation
# Works in both bull and bear markets: buys oversold deviations in downtrends, sells overbought in uptrends
# Uses 12h price action, daily VWAP bands (1.5 sigma), volume spike confirmation, and weekly EMA trend filter
# Target: 15-30 trades/year (60-120 total) with clear entry/exit rules to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Load 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate daily VWAP and standard deviation bands
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap = np.divide(vwap_numerator, vwap_denominator, 
                     out=np.full_like(vwap_numerator, np.nan), 
                     where=vwap_denominator!=0)
    
    # Calculate VWAP deviation and rolling standard deviation
    vwap_dev = typical_price_1d - vwap
    # Use 20-period rolling std dev of VWAP deviation
    vwap_dev_series = pd.Series(vwap_dev)
    vwap_std = vwap_dev_series.rolling(window=20, min_periods=20).std().values
    
    # Upper and lower VWAP bands (1.5 sigma)
    vwap_upper = vwap + 1.5 * vwap_std
    vwap_lower = vwap - 1.5 * vwap_std
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 12h timeframe
    vwap_upper_aligned = align_htf_to_ltf(prices, df_1d, vwap_upper)
    vwap_lower_aligned = align_htf_to_ltf(prices, df_1d, vwap_lower)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_upper_aligned[i]) or np.isnan(vwap_lower_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: price touches or goes below VWAP lower band + downtrend + volume spike
        if (low[i] <= vwap_lower_aligned[i] and 
            close[i] < ema50_1w_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price touches or goes above VWAP upper band + uptrend + volume spike
        elif (high[i] >= vwap_upper_aligned[i] and 
              close[i] > ema50_1w_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price returns to VWAP (mean reversion complete)
        elif position == 1 and close[i] >= vwap_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= vwap_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

# VWAP needs to be calculated and aligned
def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Load 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate daily VWAP and standard deviation bands
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap = np.divide(vwap_numerator, vwap_denominator, 
                     out=np.full_like(vwap_numerator, np.nan), 
                     where=vwap_denominator!=0)
    
    # Calculate VWAP deviation and rolling standard deviation
    vwap_dev = typical_price_1d - vwap
    # Use 20-period rolling std dev of VWAP deviation
    vwap_dev_series = pd.Series(vwap_dev)
    vwap_std = vwap_dev_series.rolling(window=20, min_periods=20).std().values
    
    # Upper and lower VWAP bands (1.5 sigma)
    vwap_upper = vwap + 1.5 * vwap_std
    vwap_lower = vwap - 1.5 * vwap_std
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 12h timeframe
    vwap_upper_aligned = align_htf_to_ltf(prices, df_1d, vwap_upper)
    vwap_lower_aligned = align_htf_to_ltf(prices, df_1d, vwap_lower)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_upper_aligned[i]) or np.isnan(vwap_lower_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: price touches or goes below VWAP lower band + downtrend + volume spike
        if (low[i] <= vwap_lower_aligned[i] and 
            close[i] < ema50_1w_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price touches or goes above VWAP upper band + uptrend + volume spike
        elif (high[i] >= vwap_upper_aligned[i] and 
              close[i] > ema50_1w_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price returns to VWAP (mean reversion complete)
        elif position == 1 and close[i] >= vwap_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= vwap_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_VWAP_Bands_1dVolume_1wEMA_MeanReversion"
timeframe = "12h"
leverage = 1.0