# 6h_WeeklyPivot_1dEMA50_Volume
# Hypothesis: Weekly pivot points (PP, R1, S1) from previous week combined with 1d EMA50 trend and volume spikes.
# Long when price breaks above weekly R1 with volume and above 1d EMA50.
# Short when price breaks below weekly S1 with volume and below 1d EMA50.
# Exits when price crosses weekly PP or reverses against EMA50.
# Designed for 6h timeframe to capture institutional breakouts with low turnover (target: 12-37 trades/year).
# Works in bull markets (breakout momentum) and bear markets (mean reversion via weekly pivot rejection).

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
    
    # Get weekly data for pivot calculation (using 1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (PP, R1, S1) from previous week
    # PP = (H + L + C) / 3
    # R1 = (2 * PP) - L
    # S1 = (2 * PP) - H
    weekly_pp = (high_1w + low_1w + close_1w) / 3
    weekly_r1 = (2 * weekly_pp) - low_1w
    weekly_s1 = (2 * weekly_pp) - high_1w
    
    # Get daily data for EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly and daily indicators to 6h timeframe
    weekly_pp_6h = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average (strict to reduce trades)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pp_6h[i]) or 
            np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or 
            np.isnan(ema50_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema50_6h[i]
        price_below_ema = close[i] < ema50_6h[i]
        
        # Price relative to weekly pivot levels
        price_above_r1 = close[i] > weekly_r1_6h[i]
        price_below_s1 = close[i] < weekly_s1_6h[i]
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume and above 1d EMA50
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume and below 1d EMA50
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly PP OR below 1d EMA50
            if (close[i] < weekly_pp_6h[i]) or (close[i] < ema50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly PP OR above 1d EMA50
            if (close[i] > weekly_pp_6h[i]) or (close[i] > ema50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0