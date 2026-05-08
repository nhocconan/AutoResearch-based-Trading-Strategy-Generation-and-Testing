#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout with 1d Trend and Volume Confirmation
# Uses daily Williams fractals to identify support/resistance levels.
# Enters on breakout above bearish fractal (resistance) or below bullish fractal (support)
# when aligned with daily trend (EMA50) and confirmed by volume spike (>1.5x 20-period average).
# Designed to capture momentum bursts in both bull and bear markets while avoiding false breakouts.
# Target: 15-35 trades/year.

name = "6h_WilliamsFractal_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams fractals and trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Calculate Williams fractals (5-bar window: 2 left, 2 right)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    bearish_fractal = np.full(len(high_daily), np.nan)  # resistance
    bullish_fractal = np.full(len(low_daily), np.nan)   # support
    
    # Need at least 5 bars: indices 2 to n-3 (0-based)
    for i in range(2, len(high_daily) - 2):
        # Bearish fractal: highest high with two lower highs on each side
        if (high_daily[i] > high_daily[i-1] and high_daily[i] > high_daily[i-2] and
            high_daily[i] > high_daily[i+1] and high_daily[i] > high_daily[i+2]):
            bearish_fractal[i] = high_daily[i]
        
        # Bullish fractal: lowest low with two higher lows on each side
        if (low_daily[i] < low_daily[i-1] and low_daily[i] < low_daily[i-2] and
            low_daily[i] < low_daily[i+1] and low_daily[i] < low_daily[i+2]):
            bullish_fractal[i] = low_daily[i]
    
    # Calculate daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 50:
        # Use pandas EMA for efficiency and correctness
        ema_series = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean()
        ema50_daily = ema_series.values
    
    # Calculate daily volume average for volume confirmation
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        vol_series = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean()
        vol_avg_20_daily = vol_series.values
    
    # Williams fractals need 2 extra bars for confirmation (wait for 2 candles after the fractal)
    bearish_fractal_confirmed = np.full_like(bearish_fractal, np.nan)
    bullish_fractal_confirmed = np.full_like(bullish_fractal, np.nan)
    
    for i in range(2, len(bearish_fractal)):
        if not np.isnan(bearish_fractal[i-2]):  # fractal formed 2 bars ago
            bearish_fractal_confirmed[i] = bearish_fractal[i-2]
        if not np.isnan(bullish_fractal[i-2]):  # fractal formed 2 bars ago
            bullish_fractal_confirmed[i] = bullish_fractal[i-2]
    
    # Align daily indicators to 6h timeframe
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_daily, bearish_fractal_confirmed)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_daily, bullish_fractal_confirmed)
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema50_daily_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume breakout: current 6h volume > 1.5x 20-period average of daily volume
        vol_breakout = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: breakout of fractal level in direction of daily trend
            # Bearish fractal = resistance, break above = bullish signal
            # Bullish fractal = support, break below = bearish signal
            
            # Long when price breaks above bearish fractal (resistance) in uptrend
            long_condition = (
                close[i] > bearish_fractal_aligned[i] and   # break above resistance
                close[i] > ema50_daily_aligned[i] and       # price above EMA50 (uptrend)
                vol_breakout                                # volume confirmation
            )
            
            # Short when price breaks below bullish fractal (support) in downtrend
            short_condition = (
                close[i] < bullish_fractal_aligned[i] and   # break below support
                close[i] < ema50_daily_aligned[i] and       # price below EMA50 (downtrend)
                vol_breakout                                # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below EMA50 or breaks below bullish fractal (support)
            if close[i] < ema50_daily_aligned[i] or close[i] < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above EMA50 or breaks above bearish fractal (resistance)
            if close[i] > ema50_daily_aligned[i] or close[i] > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals