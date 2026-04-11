#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with weekly high/low channels + daily volume confirmation.
# Uses weekly price channels (high/low of previous week) to identify trend continuation.
# Long when price breaks above weekly high with volume confirmation, short when breaks below weekly low.
# Weekly trend filter using weekly EMA to avoid counter-trend trades.
# Designed for 12-37 trades/year to minimize fee drag while capturing major trend moves.
# Works in bull markets via breakouts and in bear markets via breakdowns.

name = "12h_1w_channel_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly high/low channels from previous week
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Previous week's values for channel calculation
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    
    # First week has no previous data
    prev_weekly_high[0] = np.nan
    prev_weekly_low[0] = np.nan
    prev_weekly_close[0] = np.nan
    
    # Calculate weekly EMA for trend filter (21-period)
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema = weekly_close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly data to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, prev_weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, prev_weekly_low)
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Load daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily average volume (20-period)
    daily_volume = df_1d['volume'].values
    vol_avg_20 = np.zeros_like(daily_volume, dtype=float)
    for i in range(19, len(daily_volume)):
        vol_avg_20[i] = np.mean(daily_volume[i-19:i+1])
    vol_avg_20[:19] = np.nan
    
    # Align daily volume average to 12h timeframe
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Breakout above weekly high (long signal)
        breakout_long = high[i] >= weekly_high_aligned[i] and vol_filter and close[i] > weekly_ema_aligned[i]
        
        # Breakdown below weekly low (short signal)
        breakout_short = low[i] <= weekly_low_aligned[i] and vol_filter and close[i] < weekly_ema_aligned[i]
        
        # Exit when price returns to weekly EMA (mean reversion within the week)
        exit_long = position == 1 and close[i] <= weekly_ema_aligned[i]
        exit_short = position == -1 and close[i] >= weekly_ema_aligned[i]
        
        # Priority: breakout/breakdown > exit > hold
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals