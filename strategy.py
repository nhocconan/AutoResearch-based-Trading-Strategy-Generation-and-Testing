#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_trend_volume_v2
Hypothesis: Daily Camarilla pivot levels act as strong support/resistance. 
Price breaking above/below R4/S4 with volume and daily trend alignment indicates strong momentum.
Trades only when price is above/below 1-day EMA200 for trend filter.
Targets 15-35 trades/year by requiring confluence of Camarilla breakout, volume, and trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d OHLC for Camarilla pivots (previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: Pivot = (H+L+C)/3
    # R4 = C + ((H-L) * 1.1/2)
    # S4 = C - ((H-L) * 1.1/2)
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align to 12h timeframe (shifted by 1 for completed bar)
    camarilla_r4_12h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_12h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1-day EMA200 for trend filter
    ema200_1d = pd.Series(prev_close).ewm(span=200, adjust=False).mean().values
    ema200_12h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 24-period SMA for volume average (2 days of 12h data)
    vol_sma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if required data not available
        if (np.isnan(camarilla_r4_12h[i]) or 
            np.isnan(camarilla_s4_12h[i]) or 
            np.isnan(ema200_12h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x average volume
        vol_confirm = volume[i] > 1.8 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S4 OR trend turns down
            if close[i] < camarilla_s4_12h[i] or close[i] < ema200_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R4 OR trend turns up
            if close[i] > camarilla_r4_12h[i] or close[i] > ema200_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above R4 + volume + uptrend
            if (close[i] > camarilla_r4_12h[i] and 
                vol_confirm and 
                close[i] > ema200_12h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 + volume + downtrend
            elif (close[i] < camarilla_s4_12h[i] and 
                  vol_confirm and 
                  close[i] < ema200_12h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals