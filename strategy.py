#!/usr/bin/env python3
"""
1d Williams Fractal Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: Daily Williams fractals identify key swing points. Breakouts above/below
recent fractals with weekly EMA50 trend alignment and volume spike capture
institutional flow in both bull and bear markets. Weekly EMA50 filter ensures
we only trade with the dominant trend, reducing whipsaws. Designed for 30-100
trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr

def compute_williams_fractals(high, low):
    """Calculate Williams fractals: bearish (up) and bullish (down)"""
    n = len(high)
    bearish = np.full(n, np.nan)  # Up fractal: high[i] is highest among i-2,i-1,i,i+1,i+2
    bullish = np.full(n, np.nan)  # Down fractal: low[i] is lowest among i-2,i-1,i,i+1,i+2
    
    for i in range(2, n-2):
        if (high[i] >= high[i-1] and high[i] >= high[i-2] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish[i] = high[i]
        if (low[i] <= low[i-1] and low[i] <= low[i-2] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Williams fractals (for entry signals)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    
    # 1w data for EMA50 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = calculate_ema(df_1w['close'].values, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for trailing stop (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start index: need enough for fractals, EMA, volume MA, ATR
    start_idx = max(50, 20, 14) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal[i]) or np.isnan(bullish_fractal[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions using recent fractals
        # For long: break above most recent bearish (up) fractal
        # For short: break below most recent bullish (down) fractal
        recent_bearish = bearish_fractal[i]  # Most recent completed up fractal
        recent_bullish = bullish_fractal[i]  # Most recent completed down fractal
        
        breakout_long = curr_close > recent_bearish  # Break above recent up fractal
        breakout_short = curr_close < recent_bullish  # Break below recent down fractal
        
        if position == 0:
            # Look for entry signals - require: Fractal breakout + volume spike + 1w EMA50 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_50_1w_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_50_1w_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high_since_entry = curr_high
                lowest_low_since_entry = curr_low
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_high_since_entry = curr_high
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: update highest high and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: retrace to recent bullish fractal, trend change, or ATR trailing stop
            trailing_stop = highest_high_since_entry - 2.5 * atr[i]
            if curr_close < recent_bullish or curr_close < ema_50_1w_aligned[i] or curr_close < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest low and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: retrace to recent bearish fractal, trend change, or ATR trailing stop
            trailing_stop = lowest_low_since_entry + 2.5 * atr[i]
            if curr_close > recent_bearish or curr_close > ema_50_1w_aligned[i] or curr_close > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wEMA50_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "1d"
leverage = 1.0