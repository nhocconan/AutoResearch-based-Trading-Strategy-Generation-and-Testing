#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price crossing 1d VWAP with 1w trend filter and volume confirmation.
# Long when price crosses above 1d VWAP from below with 1w EMA50 uptrend and volume > 1.5x average.
# Short when price crosses below 1d VWAP from above with 1w EMA50 downtrend and volume > 1.5x average.
# Exit when price crosses back through 1d VWAP.
# Uses 1d VWAP as dynamic support/resistance and 1w EMA50 for trend filter to avoid counter-trend trades.
# Targets 50-150 total trades over 4 years (12-37/year) with position size 0.25.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP (typical price * volume / cumulative volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = np.divide(vwap_numerator, vwap_denominator, 
                        out=np.full_like(vwap_numerator, np.nan), 
                        where=vwap_denominator!=0)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Align 1d VWAP and 1w EMA50 to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need VWAP, EMA50, and volume MA20
    start_idx = max(19, 0)  # VWAP needs at least 1 bar, volume MA20 needs 19
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_1d_aligned[i]
        ema_trend = ema_1w_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price crosses above VWAP from below with 1w EMA50 uptrend and volume filter
            if (close[i-1] <= vwap_1d_aligned[i-1] and price > vwap and 
                price > ema_trend and vol_filter):
                signals[i] = size
                position = 1
            # Short: price crosses below VWAP from above with 1w EMA50 downtrend and volume filter
            elif (close[i-1] >= vwap_1d_aligned[i-1] and price < vwap and 
                  price < ema_trend and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below VWAP from above
            if close[i-1] >= vwap_1d_aligned[i-1] and price < vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above VWAP from below
            if close[i-1] <= vwap_1d_aligned[i-1] and price > vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_VWAP_Cross_1wEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0