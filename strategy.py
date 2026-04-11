#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_trend_atrstop_v1
# Strategy: 4h Camarilla pivot breakout with 1d trend filter, volume confirmation, and ATR stop
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (from 1d) act as strong support/resistance. Breakouts
# with volume confirmation and 1d trend alignment capture sustained moves. ATR-based stop
# limits losses. Works in bull by catching breakouts in uptrend, and in bear by catching
# breakdowns in downtrend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_trend_atrstop_v1"
timeframe = "4h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ATR (14-period) for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day's close (using 1d data)
    # Camarilla: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # We'll use the previous day's range to calculate levels for current 4h bars
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate camarilla levels for each 1d bar
    camarilla_high = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_low = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align camarilla levels to 4h timeframe (using previous day's levels)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i]) or np.isnan(atr[i])):
            # Hold current position or flat if invalid
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        # Current price and levels
        price_now = close[i]
        camarilla_high_now = camarilla_high_aligned[i]
        camarilla_low_now = camarilla_low_aligned[i]
        ema_50_now = ema_50_1d_aligned[i]
        vol_spike_now = vol_spike[i]
        
        # Breakout conditions
        bull_breakout = price_now > camarilla_high_now and vol_spike_now
        bear_breakout = price_now < camarilla_low_now and vol_spike_now
        
        # Trend filter: only trade in direction of 1d EMA50
        bull_trend = price_now > ema_50_now
        bear_trend = price_now < ema_50_now
        
        # ATR-based stop loss
        stop_long = position == 1 and price_now < entry_price - 2.0 * atr[i]
        stop_short = position == -1 and price_now > entry_price + 2.0 * atr[i]
        
        # Trading logic
        if bull_breakout and bull_trend and position != 1:
            # Enter long
            position = 1
            entry_price = price_now
            signals[i] = 0.25
        elif bear_breakout and bear_trend and position != -1:
            # Enter short
            position = -1
            entry_price = price_now
            signals[i] = -0.25
        elif stop_long or stop_short:
            # Stop loss hit
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals