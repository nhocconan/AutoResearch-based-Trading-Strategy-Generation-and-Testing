#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with weekly EMA50 trend filter and volume confirmation.
# Williams Fractals identify swing points: bearish fractal (high surrounded by two lower highs),
# bullish fractal (low surrounded by two higher lows). Breakouts above bearish fractal or below
# bullish fractal with trend confirmation (price > weekly EMA50 for long, < for short) and
# volume > 1.5x 20-period average capture momentum shifts. Weekly EMA50 filters chop and
# ensures trades align with multi-week trend, working in both bull (breakouts with trend) and
# bear (fades against weak trend via fractal failure) markets. Target: 15-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Fractals on weekly data
    # Bearish fractal: high[n] > high[n-2] and high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-2] and low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n+2]
    n_1w = len(high_1w)
    bearish_fractal = np.full(n_1w, np.nan)
    bullish_fractal = np.full(n_1w, np.nan)
    
    for i in range(2, n_1w - 2):
        if (high_1w[i] > high_1w[i-2] and high_1w[i] > high_1w[i-1] and 
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = high_1w[i]
        if (low_1w[i] < low_1w[i-2] and low_1w[i] < low_1w[i-1] and 
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = low_1w[i]
    
    # Forward fill fractal levels to maintain until next fractal
    bearish_fractal_filled = pd.Series(bearish_fractal).ffill().bfill().values
    bullish_fractal_filled = pd.Series(bullish_fractal).ffill().bfill().values
    
    # Align fractal levels to 6h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal_filled, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal_filled, additional_delay_bars=2)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or \
           np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = price > ema_50_1w_aligned[i]
        price_below_ema = price < ema_50_1w_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above bearish fractal resistance in uptrend
                if price > bearish_fractal_aligned[i] and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below bullish fractal support in downtrend
                elif price < bullish_fractal_aligned[i] and price_below_ema:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below bullish fractal support (failed hold) or trend weakens
                if price < bullish_fractal_aligned[i] or not price_above_ema:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above bearish fractal resistance (failed hold) or trend weakens
                if price > bearish_fractal_aligned[i] or not price_below_ema:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_1wEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0