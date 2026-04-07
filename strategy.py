#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v1
Hypothesis: On daily timeframe, use Donchian channel (20) breakouts for trend following, filtered by weekly trend (EMA20 > EMA50) and volume confirmation.
Enter long when price breaks above Donchian upper (20) and weekly trend is up with volume > 1.5x average.
Enter short when price breaks below Donchian lower (20) and weekly trend is down with volume > 1.5x average.
Exit when price crosses the Donchian midpoint (10-day average of high/low) or trend reverses.
Targets 7-25 trades/year to minimize fee drag while capturing sustained moves.
Works in bull markets via breakouts and bear markets via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20) - upper and lower bands
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA20 and EMA50 on weekly close
    weekly_close = df_1w['close'].values
    weekly_close_s = pd.Series(weekly_close)
    ema20_1w = weekly_close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50_1w = weekly_close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align to daily timeframe (shifted by 1 week to avoid look-ahead)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter from weekly: up if EMA20 > EMA50, down if EMA20 < EMA50
        trend_up = ema20_1w_aligned[i] > ema50_1w_aligned[i]
        trend_down = ema20_1w_aligned[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price crosses below Donchian midpoint
            if close[i] < donchian_mid[i]:
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
            # Exit when price crosses above Donchian midpoint
            if close[i] > donchian_mid[i]:
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
            # Long entry: price breaks above Donchian upper, weekly trend up, volume confirmation
            long_entry = (close[i] > donchian_high[i]) and (close[i-1] <= donchian_high[i-1]) and trend_up and vol_confirm
            
            # Short entry: price breaks below Donchian lower, weekly trend down, volume confirmation
            short_entry = (close[i] < donchian_low[i]) and (close[i-1] >= donchian_low[i-1]) and trend_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals