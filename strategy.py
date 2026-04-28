#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian(20) breakout with 1d trend filter (EMA50) and volume spike confirmation.
# Uses Donchian channels from 12h for major structure with 1d EMA50 for trend alignment.
# Long when price breaks above 12h upper Donchian with volume and price > 1d EMA50 (uptrend).
# Short when price breaks below 12h lower Donchian with volume and price < 1d EMA50 (downtrend).
# Volume spike (>2.0x 24-bar average) confirms breakout strength.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Target: 50-150 total trades over 4 years = 12-37/year for 6h.

name = "6h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid per-bar datetime conversion
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for Donchian calculation (stronger HTF structure)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band = highest high over last 20 periods
    upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band = lowest low over last 20 periods
    lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe (use previous 12h bar's channels)
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    
    # Calculate 6h volume spike: >2.0x 24-bar average volume (more conservative)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(volume_ma_24[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA50 direction (price above/below EMA50)
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Donchian breakout conditions with volume confirmation
        long_breakout = close[i] > upper_20_aligned[i] and volume_spike[i]
        short_breakout = close[i] < lower_20_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Donchian level or trend reversal
        long_exit = close[i] < lower_20_aligned[i] or close[i] < ema_50_1d_aligned[i]
        short_exit = close[i] > upper_20_aligned[i] or close[i] > ema_50_1d_aligned[i]
        
        # Handle entries and exits
        if long_breakout and price_above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and price_below_ema and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals