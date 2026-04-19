#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining Williams Fractal breakout with daily trend filter and volume confirmation.
# Williams Fractals identify potential reversal points. In trending markets (above/below daily EMA50),
# breaks of recent fractal highs/lows with volume continuation can capture momentum.
# Daily timeframe filter reduces whipsaw, volume confirms conviction. Designed for low trade frequency
# (target: 15-30 trades/year) to minimize fee drag in ranging markets like 2025.

name = "12h_WilliamsFractal_Breakout_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractals: need 5 bars (2 left, center, 2 right)
    # Bearish fractal: high[n-2] and high[n-1] < high[n] and high[n+1] and high[n+2] < high[n]
    # Bullish fractal: low[n-2] and low[n-1] > low[n] and low[n+1] and low[n+2] > low[n]
    n1d = len(high_1d)
    bearish_fractal = np.zeros(n1d, dtype=bool)
    bullish_fractal = np.zeros(n1d, dtype=bool)
    
    for i in range(2, n1d - 2):
        if (high_1d[i-2] < high_1d[i] and high_1d[i-1] < high_1d[i] and
            high_1d[i+1] < high_1d[i] and high_1d[i+2] < high_1d[i]):
            bearish_fractal[i] = True
        if (low_1d[i-2] > low_1d[i] and low_1d[i-1] > low_1d[i] and
            low_1d[i+1] > low_1d[i] and low_1d[i+2] > low_1d[i]):
            bullish_fractal[i] = True
    
    # Convert to price levels (only valid at fractal points)
    bearish_level = np.where(bearish_fractal, high_1d, np.nan)
    bullish_level = np.where(bullish_fractal, low_1d, np.nan)
    
    # Forward fill to get the most recent fractal level
    bearish_level = pd.Series(bearish_level).ffill().values
    bullish_level = pd.Series(bullish_level).ffill().values
    
    # Williams Fractals need 2 additional bars for confirmation (after the pattern completes)
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bearish_level, additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bullish_level, additional_delay_bars=2)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average (12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Enough for indicators to warm up
    
    for i in range(start_idx, n):
        if np.isnan(bearish_fractal_confirmed[i]) or np.isnan(bullish_fractal_confirmed[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_trend = ema50_1d_aligned[i]
        bear_level = bearish_fractal_confirmed[i]
        bull_level = bullish_fractal_confirmed[i]
        
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: Price breaks above recent bullish fractal low (support) + above daily EMA50 + volume
            if not np.isnan(bull_level) and price > bull_level and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below recent bearish fractal high (resistance) + below daily EMA50 + volume
            elif not np.isnan(bear_level) and price < bear_level and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below bullish fractal level OR below daily EMA50
            if not np.isnan(bull_level) and price < bull_level or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price breaks above bearish fractal level OR above daily EMA50
            if not np.isnan(bear_level) and price > bear_level or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals