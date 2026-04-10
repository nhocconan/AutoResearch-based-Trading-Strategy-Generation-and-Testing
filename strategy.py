#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
# - Long when price breaks above Camarilla H3 AND 4h EMA21 > EMA50 (bullish trend) AND hour in 08-20 UTC
# - Short when price breaks below Camarilla L3 AND 4h EMA21 < EMA50 (bearish trend) AND hour in 08-20 UTC
# - Exit: opposite Camarilla breakout (L3 for long, H3 for short)
# - Uses 1h for Camarilla calculation and entry timing, 4h for trend direction
# - Session filter reduces noise trades during low-volume hours
# - Camarilla pivots provide mathematically derived support/resistance levels
# - Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
# - Works in both bull and bear markets by following 4h trend direction

name = "1h_4h_camarilla_breakout_trend_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours ONCE before loop (Rule 10 compliance)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for trend calculation (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return signals
    
    # Calculate 4h EMAs for trend filter
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute 1h Camarilla pivots (using prior period's high/low/close)
    # Camarilla levels: H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    # H3 = Close + 1.1*(High-Low)/4, L3 = Close - 1.1*(High-Low)/4
    # We'll use H3/L3 for breakouts
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    close_shift[0] = np.nan
    
    camarilla_range = high_shift - low_shift
    camarilla_h3 = close_shift + 1.1 * camarilla_range / 4.0
    camarilla_l3 = close_shift - 1.1 * camarilla_range / 4.0
    
    for i in range(1, n):  # Start after 1-bar warmup for shift
        # Skip if any required data is invalid
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Trend filter from 4h
        bullish_trend = ema_21_4h_aligned[i] > ema_50_4h_aligned[i]
        bearish_trend = ema_21_4h_aligned[i] < ema_50_4h_aligned[i]
        
        # Camarilla breakout signals
        breakout_long = close[i] > camarilla_h3[i]
        breakout_short = close[i] < camarilla_l3[i]
        
        # Exit conditions: opposite Camarilla breakout
        exit_long = close[i] < camarilla_l3[i]
        exit_short = close[i] > camarilla_h3[i]
        
        # Trading logic
        if bullish_trend and breakout_long:
            if position != 1:  # Only signal on new long entry
                position = 1
                signals[i] = 0.20
            else:
                signals[i] = 0.20
        elif bearish_trend and breakout_short:
            if position != -1:  # Only signal on new short entry
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = -0.20
        else:
            # Check for exits
            if position == 1 and exit_long:
                position = 0
                signals[i] = 0.0
            elif position == -1 and exit_short:
                position = 0
                signals[i] = 0.0
            else:
                # Maintain current position
                signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals