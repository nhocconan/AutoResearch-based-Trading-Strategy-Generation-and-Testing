#!/usr/bin/env python3
# 4h_fractal_breakout_1d_trend_volume_v7
# Hypothesis: Focus on high-probability breakouts using daily timeframe for trend (EMA20) and fractal structure (Williams Fractals with 2-bar confirmation),
# and 4h for entry with volume confirmation (>2x average) and RSI filter (40-60). 
# Trades only during 08-20 UTC to reduce noise. Position size fixed at 0.25.
# Target: 20-50 trades/year by requiring confluence of trend, fractal breakout, volume, and momentum.
# Works in bull/bear via trend filter and avoids chop with RSI range filter.

name = "4h_fractal_breakout_1d_trend_volume_v7"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (up) and bullish (down) fractals."""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n - 2):
        # Bearish fractal: high[i] is highest of 5 bars
        if high[i] >= high[i-1] and high[i] >= high[i-2] and high[i] >= high[i+1] and high[i] >= high[i+2]:
            bearish[i] = high[i]
        # Bullish fractal: low[i] is lowest of 5 bars
        if low[i] <= low[i-1] and low[i] <= low[i-2] and low[i] <= low[i+1] and low[i] <= low[i+2]:
            bullish[i] = low[i]
    
    return bearish, bullish

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get daily data for fractals and trend filter - call ONCE before loop
    df_d = get_htf_data(prices, '1d')
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Calculate daily EMA20 for trend filter
    ema20_d = pd.Series(close_d).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate daily Williams Fractals
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_d, low_d)
    # Need 2-bar confirmation for fractals (wait for 2 candles after the fractal)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 20-period average volume for 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate RSI for momentum filter
    rsi = calculate_rsi(close, 14)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(30, 20, 14, 28)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Get aligned daily indicators for current 4h bar
        ema20_val = align_htf_to_ltf(prices, df_d, ema20_d)[i]
        bearish_val = bearish_fractal_aligned[i]
        bullish_val = bullish_fractal_aligned[i]
        rsi_val = rsi[i]
        vol_ma_val = vol_ma[i]
        
        # Skip if any required data is NaN
        if (np.isnan(ema20_val) or np.isnan(vol_ma_val) or 
            volume[i] == 0 or np.isnan(rsi_val)):
            signals[i] = 0.0
            continue
        
        # Volume breakout condition: current volume > 2.0x 20-period average
        vol_breakout = volume[i] > 2.0 * vol_ma_val
        
        # Trend filter: price above/below daily EMA20
        uptrend = close[i] > ema20_val
        downtrend = close[i] < ema20_val
        
        # Momentum filter: RSI in 40-60 range for pullback entries in trend
        rsi_healthy = 40 <= rsi_val <= 60
        
        if position == 1:  # Long position
            # Exit if price breaks below bullish fractal (support)
            if not np.isnan(bullish_val) and close[i] < bullish_val:
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above bearish fractal (resistance)
            if not np.isnan(bearish_val) and close[i] > bearish_val:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above bearish fractal (resistance) with volume confirmation, uptrend, and healthy RSI
            if (not np.isnan(bearish_val) and high[i] >= bearish_val and 
                close[i] > bearish_val and vol_breakout and uptrend and rsi_healthy):
                position = 1
                signals[i] = 0.25
            # Breakout short below bullish fractal (support) with volume confirmation, downtrend, and healthy RSI
            elif (not np.isnan(bullish_val) and low[i] <= bullish_val and 
                  close[i] < bullish_val and vol_breakout and downtrend and rsi_healthy):
                position = -1
                signals[i] = -0.25
    
    return signals