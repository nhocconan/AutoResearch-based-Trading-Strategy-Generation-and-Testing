#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout with 1d EMA50 trend filter and volume confirmation
# Long when: bullish fractal break above R3 level AND price > 1d EMA50 (uptrend) AND volume > 1.8 * 20-period avg volume
# Short when: bearish fractal break below S3 level AND price < 1d EMA50 (downtrend) AND volume > 1.8 * 20-period avg volume
# Exit with ATR-based trailing stop: signal→0 when long and price < highest_high - 2.5 * ATR OR short and price > lowest_low + 2.5 * ATR
# Uses discrete sizing 0.25 to control drawdown (BTC -77% in 2022 → ~19% loss at 0.25 exposure)
# Target: 80-180 total trades over 4 years (20-45/year) for 6h timeframe
# Williams Fractals identify key swing points; breaks indicate momentum continuation with trend and volume filters

name = "6h_WilliamsFractal_Breakout_1dEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 6h data ONCE before loop for Williams Fractals and ATR
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 10:
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Williams Fractals (5-bar: 2 left, center, 2 right)
    # Bullish fractal: low[n-2] > low[n-1] and low[n-2] > low[n] and low[n-2] > low[n+1] and low[n-2] > low[n+2]
    # Bearish fractal: high[n-2] > high[n-1] and high[n-2] > high[n] and high[n-2] > high[n+1] and high[n-2] > high[n+2]
    bullish_fractal = np.zeros(len(low_6h), dtype=bool)
    bearish_fractal = np.zeros(len(high_6h), dtype=bool)
    
    for i in range(2, len(low_6h) - 2):
        if (low_6h[i-2] > low_6h[i-1] and low_6h[i-2] > low_6h[i] and 
            low_6h[i-2] > low_6h[i+1] and low_6h[i-2] > low_6h[i+2]):
            bullish_fractal[i] = True
        if (high_6h[i-2] > high_6h[i-1] and high_6h[i-2] > high_6h[i] and 
            high_6h[i-2] > high_6h[i+1] and high_6h[i-2] > high_6h[i+2]):
            bearish_fractal[i] = True
    
    # Convert to price levels (use fractal point value when active)
    bullish_fractal_level = np.where(bullish_fractal, low_6h, np.nan)
    bearish_fractal_level = np.where(bearish_fractal, high_6h, np.nan)
    
    # Forward fill to get the most recent fractal level
    bullish_fractal_level = pd.Series(bullish_fractal_level).ffill().values
    bearish_fractal_level = pd.Series(bearish_fractal_level).ffill().values
    
    # Calculate 6h ATR(14) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume_20)
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bullish_fractal_level[i]) or 
            np.isnan(bearish_fractal_level[i]) or np.isnan(atr_14[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Long: bullish fractal break above level AND uptrend AND volume spike
            if (close[i] > bullish_fractal_level[i] and 
                close[i] > ema_50_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = close[i]
            # Short: bearish fractal break below level AND downtrend AND volume spike
            elif (close[i] < bearish_fractal_level[i] and 
                  close[i] < ema_50_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = close[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, close[i])
            # Exit long: price drops below highest_high - 2.5 * ATR (trailing stop)
            if close[i] < highest_high_since_entry - 2.5 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, close[i])
            # Exit short: price rises above lowest_low + 2.5 * ATR (trailing stop)
            if close[i] > lowest_low_since_entry + 2.5 * atr_14[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals