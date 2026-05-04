#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 1d EMA50 trend filter and volume spike confirmation
# Williams Fractals identify significant swing highs/lows. A breakout above a bearish fractal
# (or below a bullish fractal) with volume confirmation (2.0x 20-period EMA) and alignment
# with 1d EMA50 trend provides high-probability continuation entries. Designed for 4h timeframe
# to target 20-50 trades/year (75-200 total over 4 years) with discrete sizing (0.30).
# Works in bull markets by buying breakouts above bearish fractals in uptrends and in bear
# markets by selling breakdowns below bullish fractals in downtrends, avoiding false breakouts
# in ranging markets via trend and volume filters.

name = "4h_WilliamsFractal_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter and Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Fractals on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Initialize fractal arrays with NaN
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    # Williams Fractal: 5-bar pattern (requires 2 bars on each side)
    # Bearish fractal: high[n] is highest among [n-2, n-1, n, n+1, n+2]
    # Bullish fractal: low[n] is lowest among [n-2, n-1, n, n+1, n+2]
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal (swing high)
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal (swing low)
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Align fractals to 4h timeframe with additional delay for confirmation
    # Williams fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: 2.0x 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: close breaks above bearish fractal + volume confirmation + price above 1d EMA50 (uptrend)
            if (close[i] > bearish_fractal_aligned[i] and volume_confirmed and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: close breaks below bullish fractal + volume confirmation + price below 1d EMA50 (downtrend)
            elif (close[i] < bullish_fractal_aligned[i] and volume_confirmed and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price falls below bullish fractal (mean reversion) OR below 1d EMA50 (trend change)
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above bearish fractal (mean reversion) OR above 1d EMA50 (trend change)
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals