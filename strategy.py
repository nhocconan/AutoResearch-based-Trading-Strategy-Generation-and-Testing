#!/usr/bin/env python3
"""
12h_donchian_20_1d_trend_volume_v1
Hypothesis: On 12h timeframe, use Donchian channel (20) breakout for entry with 1d trend filter (EMA50 > EMA200) and volume confirmation. 
Exit on opposite Donchian breakout or trend reversal. Targets 12-37 trades/year to minimize fee drift in ranging 2025 market.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_20_1d_trend_volume_v1"
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
    
    # Donchian channel (20-period) on 12h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 and EMA200 on daily close
    daily_close = df_1d['close'].values
    daily_close_s = pd.Series(daily_close)
    ema50_1d = daily_close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = daily_close_s.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align to 12h timeframe (shifted by 1 day to avoid look-ahead)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian20 warmup
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter from 1d: up if EMA50 > EMA200, down if EMA50 < EMA200
        trend_up = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        trend_down = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on Donchian low break
            if close[i] <= donchian_low[i]:
                exit_long = True
            # Exit on trend reversal
            elif not trend_up:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on Donchian high break
            if close[i] >= donchian_high[i]:
                exit_short = True
            # Exit on trend reversal
            elif not trend_down:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price > Donchian high, 1d trend up, volume confirmation
            long_entry = (close[i] > donchian_high[i]) and trend_up and vol_confirm
            
            # Short entry: price < Donchian low, 1d trend down, volume confirmation
            short_entry = (close[i] < donchian_low[i]) and trend_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals