#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Williams Fractal for trend direction and 1d RSI for momentum confirmation.
# Long when bullish fractal confirmed on 4h (price above last bullish fractal) and 1d RSI < 40 (oversold bounce).
# Short when bearish fractal confirmed on 4h (price below last bearish fractal) and 1d RSI > 60 (overbought rejection).
# Uses tight fractal confirmation + RSI extremes to limit trades (<30/year) and avoid fee drag.
# Session filter (08-20 UTC) reduces noise. Position size 0.20.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for Williams Fractal
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Williams Fractals (requires 5-bar window: 2 left, 2 right)
    bearish_fractal = np.full(len(high_4h), np.nan)  # High with two lower highs on each side
    bullish_fractal = np.full(len(low_4h), np.nan)   # Low with two higher lows on each side
    
    for i in range(2, len(high_4h) - 2):
        # Bearish fractal: middle high is highest of 5
        if (high_4h[i] > high_4h[i-1] and high_4h[i] > high_4h[i-2] and
            high_4h[i] > high_4h[i+1] and high_4h[i] > high_4h[i+2]):
            bearish_fractal[i] = high_4h[i]
        # Bullish fractal: middle low is lowest of 5
        if (low_4h[i] < low_4h[i-1] and low_4h[i] < low_4h[i-2] and
            low_4h[i] < low_4h[i+1] and low_4h[i] < low_4h[i+2]):
            bullish_fractal[i] = low_4h[i]
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 4h fractals to 1h (need 2-bar confirmation delay for fractals)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_4h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_4h, bullish_fractal, additional_delay_bars=2)
    
    # Align 1d RSI to 1h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Pre-calculate session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need fractals and RSI
    start_idx = 50  # sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if outside session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bull_fract = bullish_fractal_aligned[i]
        bear_fract = bearish_fractal_aligned[i]
        rsi = rsi_1d_aligned[i]
        
        if position == 0:
            # Long: price above bullish fractal support AND RSI oversold (<40)
            if not np.isnan(bull_fract) and price > bull_fract and rsi < 40:
                signals[i] = size
                position = 1
            # Short: price below bearish fractal resistance AND RSI overbought (>60)
            elif not np.isnan(bear_fract) and price < bear_fract and rsi > 60:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below bullish fractal OR RSI overbought (>70)
            if (not np.isnan(bull_fract) and price < bull_fract) or rsi > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price above bearish fractal OR RSI oversold (<30)
            if (not np.isnan(bear_fract) and price > bear_fract) or rsi < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_WilliamsFractal_RSI_Confirmation"
timeframe = "1h"
leverage = 1.0