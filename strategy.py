#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day's data
    # Camarilla levels: H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    # We use the previous day's H, L, C to calculate today's levels
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    camarilla_H4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_L4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Volume filter: current volume > 1.5x 20-period average (for 4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~6 hours for 4h to reduce trades
    
    start_idx = max(100, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_H4_aligned[i]) or 
            np.isnan(camarilla_L4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction
        trend_up = close > ema_34_1d_aligned[i]
        trend_down = close < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above Camarilla H4 with volume in uptrend
            if (close[i] > camarilla_H4_aligned[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Camarilla L4 with volume in downtrend
            elif (close[i] < camarilla_L4_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below Camarilla L4 or trend changes
            if close[i] < camarilla_L4_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above Camarilla H4 or trend changes
            if close[i] > camarilla_H4_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation on 4h timeframe.
# Long when price breaks above Camarilla H4 in uptrend with volume confirmation.
# Short when price breaks below Camarilla L4 in downtrend with volume confirmation.
# Uses 4h timeframe to balance trade frequency and capture meaningful trends.
# Target: 75-200 total trades over 4 years (19-50/year) as per experiment guidelines.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Based on top-performing pattern from DB: Camarilla breakout + volume + trend filter.