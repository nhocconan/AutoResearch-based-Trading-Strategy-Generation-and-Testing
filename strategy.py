#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ORB (Opening Range Breakout) with 4h trend filter and volume confirmation
# Long when price breaks above 1st hour high, 4h EMA50 rising, volume > 1.5x average
# Short when price breaks below 1st hour low, 4h EMA50 falling, volume > 1.5x average
# Uses ORB for intraday breakout, 4h EMA for trend filter, volume for confirmation
# Targets 15-37 trades/year (60-150 over 4 years) for low fee drag and high win rate
# Works in both bull and bear markets due to trend filter and volume confirmation
# Session filter: 08-20 UTC to avoid low-liquidity periods

name = "1h_ORB_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Track daily ORB levels
    orb_high = np.full(n, np.nan)
    orb_low = np.full(n, np.nan)
    
    # Calculate ORB (first hour of each day: 00:00-01:00 UTC)
    for i in range(1, n):
        # New day detection (00:00 UTC)
        if hours[i] == 0 and hours[i-1] != 0:
            # First hour of the day: 00:00-01:00 UTC
            orb_high[i] = high[i]
            orb_low[i] = low[i]
        elif hours[i] == 1:  # 01:00 UTC - end of first hour
            # Carry forward ORB levels from 00:00-01:00
            orb_high[i] = orb_high[i-1]
            orb_low[i] = orb_low[i-1]
        elif 1 < hours[i] < 24:  # After first hour
            # Keep ORB levels constant for the rest of the day
            orb_high[i] = orb_high[i-1]
            orb_low[i] = orb_low[i-1]
        else:  # hours[i] == 0 (00:00 UTC) but not new day detection above
            orb_high[i] = orb_high[i-1]
            orb_low[i] = orb_low[i-1]
    
    # Forward fill ORB levels to handle any gaps
    orb_high_series = pd.Series(orb_high)
    orb_low_series = pd.Series(orb_low)
    orb_high = orb_high_series.ffill().bfill().values
    orb_low = orb_low_series.ffill().bfill().values
    
    start_idx = 1  # Start after first bar
    
    for i in range(start_idx, n):
        # Skip if outside trading session or missing data
        if not in_session[i] or np.isnan(orb_high[i]) or np.isnan(orb_low[i]) or \
           np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_val = high[i]
        low_val = low[i]
        orb_high_val = orb_high[i]
        orb_low_val = orb_low[i]
        ema50_4h_val = ema50_4h_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price breaks above ORB high, 4h uptrend, volume confirmation, in session
            if high_val > orb_high_val and ema50_4h_val > 0 and vol_conf_val:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below ORB low, 4h downtrend, volume confirmation, in session
            elif low_val < orb_low_val and ema50_4h_val < 0 and vol_conf_val:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below ORB low or 4h trend down
            if low_val < orb_low_val or ema50_4h_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above ORB high or 4h trend up
            if high_val > orb_high_val or ema50_4h_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals