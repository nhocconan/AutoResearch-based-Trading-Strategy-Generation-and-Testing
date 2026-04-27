#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams Fractal breakout with daily trend filter and volume confirmation.
In bear markets, price often makes lower highs (bearish fractals) before continuing down.
In bull markets, higher lows (bullish fractals) precede continuation up.
Using daily trend (EMA50) filters direction, volume confirms breakout strength.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def williams_fractals(high, low):
    """Williams Fractals: bearish = high[n-2] < high[n] and high[n-1] < high[n] and high[n+1] < high[n] and high[n+2] < high[n]; bullish inverse"""
    n = len(high)
    bearish = np.zeros(n, dtype=bool)
    bullish = np.zeros(n, dtype=bool)
    for i in range(2, n-2):
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = True
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = True
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_ltf_to_htf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Fractals on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = williams_fractals(high_1d, low_1d)
    # Fractals need 2-bar confirmation after the center bar
    bearish_aligned = align_ltf_to_htf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_aligned = align_ltf_to_htf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # Get 6-hour data for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6-hour volume MA(20)
    vol_6h = df_6h['volume'].values
    vol_ma_20_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_6h_aligned = align_ltf_to_htf(prices, df_6h, vol_ma_20_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need daily EMA, fractals, and 6h volume MA
    start_idx = max(50, 50, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or
            np.isnan(vol_ma_20_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_50 = ema_50_1d_aligned[i]
        bearish = bearish_aligned[i] > 0.5
        bullish = bullish_aligned[i] > 0.5
        vol_now = volume[i]
        vol_ma = vol_ma_20_6h_aligned[i]
        
        # Volume filter: volume > 1.5x 6h average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Williams Fractal breakout with volume and daily trend alignment
        if position == 0:
            # Long: bullish fractal + volume + daily uptrend
            if bullish and vol_filter and close[i] > ema_50:
                signals[i] = size
                position = 1
            # Short: bearish fractal + volume + daily downtrend
            elif bearish and vol_filter and close[i] < ema_50:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below daily EMA or opposite fractal appears
            if close[i] < ema_50 or bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above daily EMA or opposite fractal appears
            if close[i] > ema_50 or bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_Breakout_DailyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0