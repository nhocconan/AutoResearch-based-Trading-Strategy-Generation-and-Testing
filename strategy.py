#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1w EMA50 trend filter and volume confirmation
# Uses 12h primary timeframe for lower trade frequency (target: 50-150 total trades over 4 years)
# Williams Fractals from 1w provide strong swing high/low levels derived from weekly structure
# 1w EMA50 trend filter ensures alignment with higher timeframe momentum, effective in bull/bear regimes
# Volume spike (2.0x 20-period average) confirms institutional participation, reducing false breakouts
# Designed with tight entry conditions to minimize fee drag while maintaining edge
# Target: 75-125 total trades over 4 years (19-31/year) - within proven winning range for 12h

name = "12h_WilliamsFractal_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Williams Fractals from 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Williams Fractal: bearish (swing high) when high[n] is highest of 5 bars (n-2 to n+2)
    # bullish (swing low) when low[n] is lowest of 5 bars (n-2 to n+2)
    bearish_fractal = np.full(len(high_1w), np.nan)
    bullish_fractal = np.full(len(low_1w), np.nan)
    
    for i in range(2, len(high_1w) - 2):
        if (high_1w[i] > high_1w[i-1] and high_1w[i] > high_1w[i-2] and 
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = high_1w[i]  # Swing high (resistance)
        if (low_1w[i] < low_1w[i-1] and low_1w[i] < low_1w[i-2] and 
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = low_1w[i]   # Swing low (support)
    
    # Williams fractals need 2 extra 1w bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Calculate volume spike (2.0x 20-period average) - balanced threshold
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for volume MA and HTF data alignment)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above bullish fractal (support turned resistance) + price > 1w EMA50 + volume spike
            if close[i] > bullish_fractal_aligned[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below bearish fractal (resistance turned support) + price < 1w EMA50 + volume spike
            elif close[i] < bearish_fractal_aligned[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below bullish fractal (support level)
            if close[i] < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above bearish fractal (resistance level)
            if close[i] > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals