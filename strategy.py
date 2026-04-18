#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Bollinger Band breakout with daily trend filter and volume confirmation on 12h timeframe.
# Uses weekly BB(20,2) to identify volatility expansion and daily EMA20 for trend direction.
# Enters on 12h breakouts above/below BB bands with volume confirmation (>1.5x 20-period volume MA).
# Designed for fewer trades (target 15-30/year) to avoid fee drag in both bull and bear markets.
# Weekly timeframe provides structural context; daily EMA filters trend; volume confirms momentum.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands (20,2) on weekly close
    sma_20_1w = np.full(len(close_1w), np.nan)
    std_20_1w = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        sma_20_1w[i] = np.mean(close_1w[i-20:i])
        std_20_1w[i] = np.std(close_1w[i-20:i])
    
    upper_bb_1w = sma_20_1w + 2 * std_20_1w
    lower_bb_1w = sma_20_1w - 2 * std_20_1w
    
    # Align weekly Bollinger Bands to 12h timeframe
    upper_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_bb_1w)
    lower_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_bb_1w)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(20) on daily close
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily EMA20 to 12h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # need weekly BB, daily EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_bb_1w_aligned[i]) or np.isnan(lower_bb_1w_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above daily EMA20 (uptrend) or below (downtrend)
        trend_up = close[i] > ema_20_1d_aligned[i]
        trend_down = close[i] < ema_20_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above weekly upper BB with volume and trend filter
            if (close[i] > upper_bb_1w_aligned[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly lower BB with volume and trend filter
            elif (close[i] < lower_bb_1w_aligned[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below weekly lower BB or reverse signal
            if close[i] < lower_bb_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly upper BB or reverse signal
            if close[i] > upper_bb_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyBB20_2_DailyEMA20_VolumeFilter"
timeframe = "12h"
leverage = 1.0