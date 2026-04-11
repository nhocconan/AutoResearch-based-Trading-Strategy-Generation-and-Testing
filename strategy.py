#!/usr/bin/env python3
"""
1d_1w_camarilla_pivot_volume_v1
Strategy: Daily Camarilla pivot breakout with volume confirmation and weekly trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses daily Camarilla pivot levels (H4/L4) for breakout entries with volume confirmation (>1.5x average volume) and filtered by weekly EMA20 trend. Designed to capture breakouts in trending markets while avoiding false breakouts in chop. Uses weekly timeframe for direction and daily only for timing. Target: 30-100 total trades over 4 years.
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
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily EMA20 for trend filter (optional, using weekly as primary)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume confirmation
    
    # Calculate Camarilla levels from previous day's data
    # Need to shift by 1 to avoid look-ahead (use previous day's OHLC)
    high_1 = np.roll(high, 1)
    low_1 = np.roll(low, 1)
    close_1 = np.roll(close, 1)
    # Set first day's values to NaN (no previous day)
    high_1[0] = np.nan
    low_1[0] = np.nan
    close_1[0] = np.nan
    
    # Calculate Camarilla levels for each day
    # H4 = Close + 1.1*(High-Low)/2
    # L4 = Close - 1.1*(High-Low)/2
    camarilla_H4 = close_1 + 1.1 * (high_1 - low_1) / 2
    camarilla_L4 = close_1 - 1.1 * (high_1 - low_1) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(camarilla_H4[i]) or np.isnan(camarilla_L4[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below weekly EMA20
        uptrend = price_close > ema_20_1w_aligned[i]
        downtrend = price_close < ema_20_1w_aligned[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = price_close > camarilla_H4[i]
        breakout_down = price_close < camarilla_L4[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend
        
        # Exit when price returns to the EMA20 (daily) or opposite Camarilla level
        exit_long = position == 1 and (price_close < ema_20[i] or price_close < camarilla_L4[i])
        exit_short = position == -1 and (price_close > ema_20[i] or price_close > camarilla_H4[i])
        
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