#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above latest bearish fractal, price > 12h EMA50, and volume > 2.0x 20-bar average
# Short when price breaks below latest bullish fractal, price < 12h EMA50, and volume > 2.0x 20-bar average
# Uses 12h EMA for higher timeframe trend alignment (matches experiment HTF)
# Volume spike confirms breakout strength
# Williams Fractals require 2-bar confirmation delay to avoid look-ahead
# Discrete position sizing (0.25) to minimize fee churn
# Designed for low trade frequency (12-37/year on 6h) to avoid fee drag
# Works in bull (breakouts above rising EMA) and bear (breakdowns below falling EMA)

name = "6h_WilliamsFractal_Breakout_12hEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 6h timeframe (wait for completed 12h bar)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams Fractals on 12h
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    n_12h = len(high_12h)
    
    bearish_fractal = np.full(n_12h, np.nan)
    bullish_fractal = np.full(n_12h, np.nan)
    
    # Calculate fractals (need at least 5 points: n-2, n-1, n, n+1, n+2)
    for i in range(2, n_12h - 2):
        # Bearish fractal: middle bar is highest
        if (high_12h[i-2] < high_12h[i-1] and 
            high_12h[i] < high_12h[i-1] and
            high_12h[i-1] > high_12h[i-3] and
            high_12h[i-1] > high_12h[i+1]):
            bearish_fractal[i-1] = high_12h[i-1]
        
        # Bullish fractal: middle bar is lowest
        if (low_12h[i-2] > low_12h[i-1] and 
            low_12h[i] > low_12h[i-1] and
            low_12h[i-1] < low_12h[i-3] and
            low_12h[i-1] < low_12h[i+1]):
            bullish_fractal[i-1] = low_12h[i-1]
    
    # Align fractals to 6h timeframe with 2-bar additional delay for confirmation
    # Williams fractals need 2 extra 12h bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_12h, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_12h, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 20) + 2  # EMA(50) + volume MA(20) + fractal delay warmup
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > bearish fractal (resistance), price > 12h EMA50, volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < bullish fractal (support), price < 12h EMA50, volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < bullish fractal (support) or price < 12h EMA50
            if (close[i] < bullish_fractal_aligned[i] or 
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > bearish fractal (resistance) or price > 12h EMA50
            if (close[i] > bearish_fractal_aligned[i] or 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals