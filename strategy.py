#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d trend filter and volume confirmation
# - Long when price breaks above latest bearish fractal AND 1d EMA21 rising AND volume > 1.5x 50-bar avg
# - Short when price breaks below latest bullish fractal AND 1d EMA21 falling AND volume > 1.5x 50-bar avg
# - Exit when price crosses 1d EMA21 (trend reversal signal)
# - Uses Williams Fractals for swing high/low detection (requires 2-bar confirmation delay)
# - 12h timeframe targets 12-37 trades/year (50-150 total over 4 years)
# - Fractals work in both ranging and trending markets; EMA filter adds trend bias

name = "12h_1d_williams_fractal_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(21) for trend filter
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Pre-compute Williams Fractals from 1d data (requires 2-bar confirmation delay)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Bearish fractal: swing high (needs 2 bars after to confirm)
    # Bullish fractal: swing low (needs 2 bars after to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Pre-compute volume confirmation: > 1.5x 50-period average
    volume_50_avg = prices['volume'].rolling(window=50, min_periods=50).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_50_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema21_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(volume_50_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above bearish fractal (swing high) AND 1d uptrend with volume spike
            if (prices['close'].iloc[i] > bearish_fractal_aligned[i] and 
                prices['close'].iloc[i] > ema21_1d_aligned[i] and  # price above 1d EMA21
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below bullish fractal (swing low) AND 1d downtrend with volume spike
            elif (prices['close'].iloc[i] < bullish_fractal_aligned[i] and 
                  prices['close'].iloc[i] < ema21_1d_aligned[i] and  # price below 1d EMA21
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on trend reversal
            # Exit when price crosses 1d EMA21 (trend change signal)
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= ema21_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= ema21_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def compute_williams_fractals(high, low):
    """Compute Williams Fractals: bearish (swing high) and bullish (swing low)"""
    n = len(high)
    bearish = np.full(n, np.nan)  # swing high
    bullish = np.full(n, np.nan)  # swing low
    
    # Need at least 5 points: 2 left, center, 2 right
    for i in range(2, n - 2):
        # Bearish fractal: high[i] is highest among i-2, i-1, i, i+1, i+2
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        
        # Bullish fractal: low[i] is lowest among i-2, i-1, i, i+1, i+2
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish