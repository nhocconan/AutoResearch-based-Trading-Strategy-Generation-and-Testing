#!/usr/bin/env python3
"""
4h_donchian_20_1d_trend_volume_v2
Hypothesis: On 4h timeframe, use Donchian channel breakout (20-period) for entry, filtered by daily trend (EMA20 > EMA50) and volume confirmation (>1.5x average). Exit on opposite Donchian breakout or trend reversal.
Targets 20-50 trades/year to minimize fee drag while capturing sustained moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_trend_volume_v2"
timeframe = "4h"
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
    
    # Donchian channel (20-period) - using high/low of last 20 periods
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
    
    # Calculate EMA20 and EMA50 on daily close
    daily_close = df_1d['close'].values
    daily_close_s = pd.Series(daily_close)
    ema20_1d = daily_close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50_1d = daily_close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align to 4h timeframe (shifted by 1 day to avoid look-ahead)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter from daily: up if EMA20 > EMA50, down if EMA20 < EMA50
        trend_up = ema20_1d_aligned[i] > ema50_1d_aligned[i]
        trend_down = ema20_1d_aligned[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price breaks below Donchian low
            if close[i] < donchian_low[i]:
                exit_long = True
            # Exit on trend reversal
            elif not trend_up:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price breaks above Donchian high
            if close[i] > donchian_high[i]:
                exit_short = True
            # Exit on trend reversal
            elif not trend_down:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high, daily trend up, volume confirmation
            long_entry = (close[i] > donchian_high[i]) and trend_up and vol_confirm
            
            # Short entry: price breaks below Donchian low, daily trend down, volume confirmation
            short_entry = (close[i] < donchian_low[i]) and trend_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.30
            elif short_entry:
                position = -1
                signals[i] = -0.30
    
    return signals