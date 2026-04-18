#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h trading using weekly Donchian channel (20) breakouts with weekly EMA trend filter and volume confirmation.
# Weekly Donchian provides structural support/resistance, EMA filters trend direction, volume confirms breakout strength.
# Designed for low trade frequency (target 15-30/year) to minimize fee drag while capturing major trend moves in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and EMA
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channel (20-period)
    upper_20 = np.full(len(high_1w), np.nan)
    lower_20 = np.full(len(low_1w), np.nan)
    for i in range(20, len(high_1w)):
        upper_20[i] = np.max(high_1w[i-20:i])
        lower_20[i] = np.min(low_1w[i-20:i])
    
    # Align weekly Donchian to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # Calculate weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 12h ATR for stop loss and position sizing
    tr_1 = high - low
    tr_2 = np.abs(high - np.roll(close, 1))
    tr_3 = np.abs(low - np.roll(close, 1))
    tr_1[0] = high[0] - low[0]
    tr_2[0] = np.abs(high[0] - close[0])
    tr_3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr_1, np.maximum(tr_2, tr_3))
    atr_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (50-period)
    vol_ma = np.full(n, np.nan)
    for i in range(50, n):
        vol_ma[i] = np.mean(volume[i-50:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(70, 50)  # need weekly Donchian (20+50), volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0 * 50-period average
        vol_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: price above/below weekly EMA34
        trend_up = close[i] > ema_34_1w_aligned[i]
        trend_down = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian upper with volume and trend filter
            if (close[i] > upper_20_aligned[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly Donchian lower with volume and trend filter
            elif (close[i] < lower_20_aligned[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below weekly Donchian lower or ATR-based stop
            if close[i] < lower_20_aligned[i] - 1.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly Donchian upper or ATR-based stop
            if close[i] > upper_20_aligned[i] + 1.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyDonchian20_EMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0