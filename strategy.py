#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above upper band with bullish 1w trend and volume spike.
Short when price breaks below lower band with bearish 1w trend and volume spike.
Exit when price returns to middle band or trend weakens.
Uses weekly EMA34 for trend filter to avoid whipsaws in ranging markets.
Designed for low trade frequency (10-25/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 35:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_w = pd.Series(df_weekly['close'].values)
    ema34_w = close_w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to daily timeframe
    ema34_w_aligned = align_htf_to_ltf(prices, df_weekly, ema34_w)
    
    # Calculate daily Donchian channels (20-period) using daily high/low
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper = high_s.rolling(window=20, min_periods=20).max().values
    lower = low_s.rolling(window=20, min_periods=20).min().values
    middle = (upper + lower) / 2.0
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(ema34_w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band with bullish 1w trend and volume spike
            if (close[i] > upper[i] and 
                close[i] > ema34_w_aligned[i] and  # Bullish trend: price above weekly EMA34
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band with bearish 1w trend and volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema34_w_aligned[i] and  # Bearish trend: price below weekly EMA34
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle OR trend turns bearish
                if close[i] <= middle[i] or close[i] < ema34_w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle OR trend turns bullish
                if close[i] >= middle[i] or close[i] > ema34_w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_20_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0