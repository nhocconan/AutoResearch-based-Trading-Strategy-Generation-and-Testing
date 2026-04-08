#!/usr/bin/env python3
# 4h_fractal_breakout_1d_trend_volume_v7
# Hypothesis: Tighten entry conditions from v6 by requiring volume > 4x average (not 3x) and adding ADX > 25 trend strength filter to avoid chop. This should reduce trades to ~15-25/year while maintaining edge in both bull and bear markets via fractal breakouts with volume and trend confirmation.

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

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    
    for i in range(1, len(high)):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
            
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
    
    tr = np.zeros_like(high)
    for i in range(len(high)):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros_like(high)
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, len(high)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    plus_di = 100 * np.where(atr != 0, 
                            np.convolve(plus_dm, np.ones(period)/period, mode='same') / atr, 0)
    minus_di = 100 * np.where(atr != 0,
                             np.convolve(minus_dm, np.ones(period)/period, mode='same') / atr, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.zeros_like(close)
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, len(close)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    
    # Calculate ADX for trend strength filter
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(30, 20, 14, 28)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Get aligned daily indicators for current 4h bar
        ema20_val = align_htf_to_ltf(prices, df_d, ema20_d)[i]
        bearish_val = bearish_fractal_aligned[i]
        bullish_val = bullish_fractal_aligned[i]
        rsi_val = rsi[i]
        vol_ma_val = vol_ma[i]
        adx_val = adx[i]
        
        # Skip if any required data is NaN
        if (np.isnan(ema20_val) or np.isnan(vol_ma_val) or np.isnan(adx_val) or 
            volume[i] == 0 or np.isnan(rsi_val)):
            signals[i] = 0.0
            continue
        
        # Volume breakout condition: current volume > 4.0x 20-period average (tighter)
        vol_breakout = volume[i] > 4.0 * vol_ma_val
        
        # Trend filter: price above/below daily EMA20
        uptrend = close[i] > ema20_val
        downtrend = close[i] < ema20_val
        
        # Momentum filter: RSI > 50 for bullish momentum, < 50 for bearish
        bullish_momentum = rsi_val > 50
        bearish_momentum = rsi_val < 50
        
        # Trend strength filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
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
            # Breakout long above bearish fractal (resistance) with volume confirmation, uptrend, bullish momentum, and strong trend
            if (not np.isnan(bearish_val) and high[i] >= bearish_val and 
                close[i] > bearish_val and vol_breakout and uptrend and bullish_momentum and strong_trend):
                position = 1
                signals[i] = 0.25
            # Breakout short below bullish fractal (support) with volume confirmation, downtrend, bearish momentum, and strong trend
            elif (not np.isnan(bullish_val) and low[i] <= bullish_val and 
                  close[i] < bullish_val and vol_breakout and downtrend and bearish_momentum and strong_trend):
                position = -1
                signals[i] = -0.25
    
    return signals