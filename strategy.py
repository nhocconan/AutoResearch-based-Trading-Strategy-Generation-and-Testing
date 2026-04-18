#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian channel breakout with daily volume confirmation and trend filter on 1d timeframe.
# Uses weekly Donchian channels (20-period) for long-term structure, daily volume > 1.5x 20-day average for confirmation,
# and daily close > weekly EMA34 for trend alignment. Designed for low frequency (target 10-25 trades/year) to minimize fee drag.
# Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume) by capturing institutional participation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and EMA
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    upper_20 = np.full(len(high_1w), np.nan)
    lower_20 = np.full(len(low_1w), np.nan)
    for i in range(20, len(high_1w)):
        upper_20[i] = np.max(high_1w[i-20:i])
        lower_20[i] = np.min(low_1w[i-20:i])
    
    # Align weekly Donchian to daily timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # Calculate weekly EMA(34) on close
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # need weekly Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-day average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: close above weekly EMA34 (uptrend) or below (downtrend)
        trend_up = close[i] > ema_34_1w_aligned[i]
        trend_down = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long entry: close above weekly upper Donchian + volume + trend
            if (close[i] > upper_20_aligned[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: close below weekly lower Donchian + volume + trend
            elif (close[i] < lower_20_aligned[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close crosses below weekly lower Donchian
            if close[i] < lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close crosses above weekly upper Donchian
            if close[i] > upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_Volume_EMA34Trend"
timeframe = "1d"
leverage = 1.0