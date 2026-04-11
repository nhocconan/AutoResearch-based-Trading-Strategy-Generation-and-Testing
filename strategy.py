#!/usr/bin/env python3
"""
1d_1w_camarilla_pivot_volume_v1
Strategy: 1d Camarilla pivot breakout with volume confirmation and 1w trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses weekly trend filter (price above/below weekly EMA20) to determine direction, then looks for breakouts of daily Camarilla H3/L3 levels with volume confirmation (>1.5x average volume). Designed to capture strong trending moves while avoiding false breakouts in choppy markets. Weekly trend ensures we only trade in the direction of the higher timeframe momentum, reducing whipsaw. Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily average volume (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume confirmation threshold
    
    # Calculate Camarilla levels from previous day's data
    # We need to shift the daily data by 1 to avoid look-ahead
    high_1 = np.roll(high, 1)
    low_1 = np.roll(low, 1)
    close_1 = np.roll(close, 1)
    # Set first value to NaN since there's no previous day
    high_1[0] = np.nan
    low_1[0] = np.nan
    close_1[0] = np.nan
    
    # Calculate Camarilla H3 and L3 levels
    camarilla_H3 = close_1 + 1.1 * (high_1 - low_1) / 4
    camarilla_L3 = close_1 - 1.1 * (high_1 - low_1) / 4
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg[i]) or
            np.isnan(camarilla_H3[i]) or np.isnan(camarilla_L3[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below weekly EMA20
        uptrend_1w = price_close > ema_20_1w_aligned[i]
        downtrend_1w = price_close < ema_20_1w_aligned[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = price_close > camarilla_H3[i]
        breakout_down = price_close < camarilla_L3[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend_1w
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend_1w
        
        # Exit when price returns to the opposite Camarilla level (H3 for shorts, L3 for longs)
        exit_long = position == 1 and price_close < camarilla_L3[i]
        exit_short = position == -1 and price_close > camarilla_H3[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals