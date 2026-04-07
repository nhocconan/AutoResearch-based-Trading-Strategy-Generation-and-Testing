#!/usr/bin/env python3
"""
12h_donchian_20_1d_trend_volume_v1
Hypothesis: On 12h timeframe, use Donchian channel breakout (20) with daily trend filter (EMA50 > EMA100) and volume confirmation.
Enter long when price breaks above upper Donchian band with bullish daily trend and volume > 1.5x average.
Enter short when price breaks below lower Donchian band with bearish daily trend and volume > 1.5x average.
Exit when price returns to Donchian middle or trend reverses.
Targets 12-37 trades/year to minimize fee drift while capturing sustained trends.
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
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    donchian_mid = (donchian_high + donchian_low) / 2
    
    donchian_high_vals = donchian_high.values
    donchian_low_vals = donchian_low.values
    donchian_mid_vals = donchian_mid.values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 and EMA100 on daily close
    daily_close = df_1d['close'].values
    daily_close_s = pd.Series(daily_close)
    ema50_1d = daily_close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema100_1d = daily_close_s.ewm(span=100, min_periods=100, adjust=False).mean().values
    
    # Align to 12h timeframe (shifted by 1 day to avoid look-ahead)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_vals[i]) or np.isnan(donchian_low_vals[i]) or 
            np.isnan(donchian_mid_vals[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema100_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter from daily: up if EMA50 > EMA100, down if EMA50 < EMA100
        trend_up = ema50_1d_aligned[i] > ema100_1d_aligned[i]
        trend_down = ema50_1d_aligned[i] < ema100_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price returns to Donchian middle
            if close[i] <= donchian_mid_vals[i]:
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
            # Exit when price returns to Donchian middle
            if close[i] >= donchian_mid_vals[i]:
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
            # Long entry: price breaks above upper Donchian band, daily trend up, volume confirmation
            long_entry = (close[i] > donchian_high_vals[i]) and (close[i-1] <= donchian_high_vals[i-1]) and trend_up and vol_confirm
            
            # Short entry: price breaks below lower Donchian band, daily trend down, volume confirmation
            short_entry = (close[i] < donchian_low_vals[i]) and (close[i-1] >= donchian_low_vals[i-1]) and trend_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals