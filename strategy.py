#!/usr/bin/env python3
# Hypothesis: 4h price crossing above/below 1-day Volume Weighted Average Price (VWAP) with 1-week EMA50 trend filter and volume confirmation
# Long when price crosses above VWAP, weekly EMA50 rising, and volume > 1.5x average
# Short when price crosses below VWAP, weekly EMA50 falling, and volume > 1.5x average
# Exit when price returns to VWAP or weekly EMA50 reverses direction
# Uses VWAP for intraday fair value, weekly EMA for trend, volume for conviction
# Designed to capture institutional flow with controlled frequency in both bull and bear markets
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "4h_VWAP_Cross_1wEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d VWAP (typical price * volume cumulative / volume cumulative)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.where(cum_vol > 0, cum_pv / cum_vol, typical_price)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price crosses above VWAP, weekly EMA50 rising, volume confirmation
            if (close[i] > vwap[i] and close[i-1] <= vwap[i-1] and  # crossed above
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and    # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below VWAP, weekly EMA50 falling, volume confirmation
            elif (close[i] < vwap[i] and close[i-1] >= vwap[i-1] and  # crossed below
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and    # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to VWAP or weekly EMA50 starts falling
            if (close[i] <= vwap[i]) or (ema50_1w_aligned[i] < ema50_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to VWAP or weekly EMA50 starts rising
            if (close[i] >= vwap[i]) or (ema50_1w_aligned[i] > ema50_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals