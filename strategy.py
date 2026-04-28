#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout + 1w Trend Filter + Volume Spike
# Williams fractals identify significant swing highs/lows on weekly chart.
# Breakout above prior weekly bearish fractal with 1w uptrend and volume spike = long.
# Breakdown below prior weekly bullish fractal with 1w downtrend and volume spike = short.
# Weekly fractals require 2-bar confirmation delay (additional_delay_bars=2).
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull/bear markets by requiring alignment with 1w trend.
# Volume confirmation filters weak breakouts.

name = "6h_WilliamsFractal_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for fractal calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 5:  # Need at least 5 bars for Williams fractals
        return np.zeros(n)
    
    # Calculate Williams fractals on weekly data
    # Bearish fractal: high[n-2] > high[n-3] and high[n-2] > high[n-1] and high[n-2] > high[n] and high[n-2] > high[n+1]
    # Bullish fractal: low[n-2] < low[n-3] and low[n-2] < low[n-1] and low[n-2] < low[n] and low[n-2] < low[n+1]
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    bearish_fractal = np.full(len(high_1w), np.nan)
    bullish_fractal = np.full(len(low_1w), np.nan)
    
    for i in range(2, len(high_1w) - 2):
        # Bearish fractal (swing high)
        if (high_1w[i] > high_1w[i-2] and high_1w[i] > high_1w[i-1] and 
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = high_1w[i]
        # Bullish fractal (swing low)
        if (low_1w[i] < low_1w[i-2] and low_1w[i] < low_1w[i-1] and 
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = low_1w[i]
    
    # Align fractals to 6h with 2-bar confirmation delay (needed for fractal confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure sufficient history for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1w EMA trend filter
        ema_trend_up = close[i] > ema_34_1w_aligned[i]
        ema_trend_down = close[i] < ema_34_1w_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > weekly bearish fractal, 1w EMA34 uptrend, volume confirm
            if price > bearish_fractal_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Price < weekly bullish fractal, 1w EMA34 downtrend, volume confirm
            elif price < bullish_fractal_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement below bullish fractal or trend change
            if price < bullish_fractal_aligned[i] or not ema_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on retracement above bearish fractal or trend change
            if price > bearish_fractal_aligned[i] or not ema_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals