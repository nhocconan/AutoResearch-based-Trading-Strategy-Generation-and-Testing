#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour price position within daily Bollinger Bands combined with weekly trend filter
# We go long when price is in the lower 20% of daily Bollinger Bands (oversold) with weekly EMA(50) uptrend
# We go short when price is in the upper 20% of daily Bollinger Bands (overbought) with weekly EMA(50) downtrend
# Uses mean reversion in ranging markets and trend following in trending markets via weekly filter
# Targets 12-37 trades per year on 6H timeframe to avoid excessive trading
# Bollinger Bands provide dynamic support/resistance that adapts to volatility
# Weekly trend filter ensures we trade with higher timeframe momentum

name = "6h_BollingerPosition_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Bollinger Bands (20, 2)
    daily_close = df_1d['close'].values
    sma20 = pd.Series(daily_close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(daily_close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + (2 * std20)
    lower_band = sma20 - (2 * std20)
    
    # Align Bollinger Bands to 6H timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    sma20_aligned = align_htf_to_ltf(prices, df_1d, sma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(sma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1w_val = ema50_1w_aligned[i]
        upper_band_val = upper_band_aligned[i]
        lower_band_val = lower_band_aligned[i]
        sma20_val = sma20_aligned[i]
        close_val = close[i]
        
        if position == 0:
            # Enter long: price in lower 20% of Bollinger Bands + weekly uptrend
            band_width = upper_band_val - lower_band_val
            if band_width > 0:  # avoid division by zero
                position_in_bands = (close_val - lower_band_val) / band_width
                if position_in_bands < 0.2 and close_val > ema50_1w_val:
                    signals[i] = 0.25
                    position = 1
            # Enter short: price in upper 20% of Bollinger Bands + weekly downtrend
            elif position_in_bands > 0.8 and close_val < ema50_1w_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above SMA(20) or weekly trend turns down
            if close_val > sma20_val or close_val < ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below SMA(20) or weekly trend turns up
            if close_val < sma20_val or close_val > ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals