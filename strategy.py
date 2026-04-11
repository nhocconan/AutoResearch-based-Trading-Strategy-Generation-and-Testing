#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_volume_trend_v1
# Strategy: Daily Camarilla breakout with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Price tends to break out from Camarilla pivot levels on the daily chart with momentum.
# Uses weekly trend filter to avoid counter-trend trades and volume confirmation for institutional participation.
# Designed to work in both bull and bear markets by following the dominant weekly trend.
# Targets 30-100 trades over 4 years (7-25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_trend_v1"
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
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    # Camarilla levels: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # We use previous day's data to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # Set first day's values to NaN (no previous day)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla H4 and L4 levels
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(camarilla_h4[i]) or 
            np.isnan(camarilla_l4[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        vol = volume[i]
        
        # Trend filter: price above/below weekly EMA200
        uptrend_1w = price_close > ema_200_1w_aligned[i]
        downtrend_1w = price_close < ema_200_1w_aligned[i]
        
        # Volume confirmation: volume > 1.5 * average volume
        vol_confirm = vol > (1.5 * vol_avg[i])
        
        # Breakout signals
        long_signal = (price_close > camarilla_h4[i]) and uptrend_1w and vol_confirm
        short_signal = (price_close < camarilla_l4[i]) and downtrend_1w and vol_confirm
        
        # Exit when price returns to previous day's close (mean reversion)
        exit_long = position == 1 and (price_close < prev_close[i])
        exit_short = position == -1 and (price_close > prev_close[i])
        
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