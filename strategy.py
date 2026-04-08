#!/usr/bin/env python3
# 4h_fractal_breakout_1d_trend_volume_v12
# Hypothesis: Improve upon v11 by further tightening entry conditions to reduce trade frequency and avoid overtrading.
# Changes from v11: Increased volume confirmation threshold to 5x average, narrowed RSI range to 45-55, and added a minimum holding period of 3 bars.
# Uses Williams Fractals from daily timeframe with 2-bar confirmation, trend filter via daily EMA20, and momentum filter via RSI (45-55).
# Trades only during 08-20 UTC to avoid low-liquidity periods. Position size fixed at 0.25.
# Target: 10-25 trades/year by requiring confluence of trend, fractal breakout, high volume, healthy momentum, and low volatility.
# Designed to work in both bull and bear markets via trend filter and avoid chop with volatility filter and moderate RSI range.

name = "4h_fractal_breakout_1d_trend_volume_v12"
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
    
    # Calculate ATR for volatility filter (14-period)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0  # Track bars since entry for minimum holding period
    
    # Start from sufficient lookback
    start_idx = max(30, 20, 14, 28)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            bars_since_entry = 0  # Reset counter when out of session
            continue
        
        # Get aligned daily indicators for current 4h bar
        ema20_val = align_htf_to_ltf(prices, df_d, ema20_d)[i]
        bearish_val = bearish_fractal_aligned[i]
        bullish_val = bullish_fractal_aligned[i]
        rsi_val = rsi[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        atr_ma_val = atr_ma[i]
        
        # Skip if any required data is NaN
        if (np.isnan(ema20_val) or np.isnan(vol_ma_val) or np.isnan(atr_val) or np.isnan(atr_ma_val) or
            volume[i] == 0 or np.isnan(rsi_val)):
            signals[i] = 0.0
            bars_since_entry = 0  # Reset counter when data invalid
            continue
        
        # Volatility filter: current ATR < 1.5x 20-period average ATR (avoid choppy markets)
        vol_filter = atr_val < 1.5 * atr_ma_val
        
        # Volume breakout condition: current volume > 5.0x 20-period average (tighter)
        vol_breakout = volume[i] > 5.0 * vol_ma_val
        
        # Trend filter: price above/below daily EMA20
        uptrend = close[i] > ema20_val
        downtrend = close[i] < ema20_val
        
        # Momentum filter: RSI in narrow 45-55 range
        rsi_healthy = 45 <= rsi_val <= 55
        
        if position == 1:  # Long position
            bars_since_entry += 1
            # Exit if price breaks below bullish fractal (support) OR minimum holding period met
            if (not np.isnan(bullish_val) and close[i] < bullish_val) or bars_since_entry >= 3:
                position = 0
                signals[i] = 0.0
                bars_since_entry = 0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            bars_since_entry += 1
            # Exit if price breaks above bearish fractal (resistance) OR minimum holding period met
            if (not np.isnan(bearish_val) and close[i] > bearish_val) or bars_since_entry >= 3:
                position = 0
                signals[i] = 0.0
                bars_since_entry = 0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            bars_since_entry = 0
            # Breakout long above bearish fractal (resistance) with volume confirmation, uptrend, healthy momentum, and low volatility
            if (not np.isnan(bearish_val) and high[i] >= bearish_val and 
                close[i] > bearish_val and vol_breakout and uptrend and rsi_healthy and vol_filter):
                position = 1
                signals[i] = 0.25
            # Breakout short below bullish fractal (support) with volume confirmation, downtrend, healthy momentum, and low volatility
            elif (not np.isnan(bullish_val) and low[i] <= bullish_val and 
                  close[i] < bullish_val and vol_breakout and downtrend and rsi_healthy and vol_filter):
                position = -1
                signals[i] = -0.25
    
    return signals