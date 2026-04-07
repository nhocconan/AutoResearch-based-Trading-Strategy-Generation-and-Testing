#!/usr/bin/env python3
"""
12h_volatility_breakout_1d_trend_v1
Hypothesis: On 12h timeframe, break above/below 20-period high/low with volume > 2x average and 1d trend confirmation (close > SMA50 for long, close < SMA50 for short). 
Volatility breakouts capture momentum in both bull and bear markets, while volume confirmation and trend filter reduce false signals. Targets 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_volatility_breakout_1d_trend_v1"
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
    
    # Donchian channel (20-period high/low) on 12h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for trend filter (close vs SMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate SMA50 on daily close
    daily_close = df_1d['close'].values
    daily_close_s = pd.Series(daily_close)
    sma50_1d = daily_close_s.rolling(window=50, min_periods=50).mean().values
    
    # Align to 12h timeframe (shifted by 1 day to avoid look-ahead)
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after Donchian20 warmup
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(sma50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter from 1d: close above/below SMA50
        price_above_sma50 = close[i] > sma50_1d_aligned[i]
        price_below_sma50 = close[i] < sma50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on price breaking below Donchian low
            if close[i] < donchian_low[i]:
                exit_long = True
            # Exit when volume drops below average
            elif volume[i] < vol_ma[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on price breaking above Donchian high
            if close[i] > donchian_high[i]:
                exit_short = True
            # Exit when volume drops below average
            elif volume[i] < vol_ma[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: break above Donchian high, price above SMA50, volume confirmation
            long_entry = (close[i] > donchian_high[i]) and price_above_sma50 and vol_confirm
            
            # Short entry: break below Donchian low, price below SMA50, volume confirmation
            short_entry = (close[i] < donchian_low[i]) and price_below_sma50 and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals