#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian breakout with weekly RSI filter and volume confirmation.
Long when price breaks above 1-week Donchian high and weekly RSI > 60.
Short when price breaks below 1-week Donchian low and weekly RSI < 40.
Exit when price crosses midline (average of high/low) of the 1-week channel.
Weekly RSI filters trend strength, Donchian provides breakout signals, volume confirms institutional interest.
Works in bull markets via breakouts and bear via shorting breakdowns with RSI preventing counter-trend entries.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-week data for Donchian and RSI - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Weekly RSI (14-period)
    delta = pd.Series(df_1w['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = rsi_1w.fillna(50).values  # neutral when undefined
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high, RSI > 60, volume above average
            if (close[i] > donchian_high_aligned[i] and 
                rsi_1w_aligned[i] > 60 and 
                volume[i] > vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low, RSI < 40, volume above average
            elif (close[i] < donchian_low_aligned[i] and 
                  rsi_1w_aligned[i] < 40 and 
                  volume[i] > vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below weekly Donchian midline
                if close[i] < donchian_mid_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above weekly Donchian midline
                if close[i] > donchian_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_WeeklyRSI_Volume"
timeframe = "1d"
leverage = 1.0