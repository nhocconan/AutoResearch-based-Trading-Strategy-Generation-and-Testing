#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout + Weekly Trend Filter + Volume Confirmation
# - Primary signal: Williams Fractal breakout - price breaks above latest bearish fractal (resistance) for long, below bullish fractal (support) for short
# - Trend filter: 1w EMA50 - price must be above EMA for longs, below for shorts (higher timeframe alignment)
# - Volume confirmation: 6h volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Fractals identify key support/resistance, weekly EMA ensures alignment with higher timeframe trend

name = "6h_1w_williams_fractal_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute Williams Fractals on 1d timeframe (for better signal quality)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractals: 5-point pattern (requires 2 bars on each side)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] < high[n+1] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] > low[n+1] < low[n+2]
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        # Bearish fractal (resistance)
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i+1] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal (support)
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i+1] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Forward fill to get the most recent fractal level
    bearish_fractal = pd.Series(bearish_fractal).ffill().values
    bullish_fractal = pd.Series(bullish_fractal).ffill().values
    
    # Align fractals to primary timeframe (completed 1d bar only, +2 extra bars for confirmation)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(bearish_aligned[i]) or
            np.isnan(bullish_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA50 OR price drops below bullish fractal (support broken)
            if close[i] < ema_50_aligned[i] or close[i] < bullish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA50 OR price rises above bearish fractal (resistance broken)
            if close[i] > ema_50_aligned[i] or close[i] > bearish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for fractal breakout with volume confirmation and weekly EMA50 filter
            # Long: price breaks above bearish fractal (resistance) AND volume regime AND price above weekly EMA50
            if (close[i] > bearish_aligned[i] and 
                volume_regime[i] and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below bullish fractal (support) AND volume regime AND price below weekly EMA50
            elif (close[i] < bullish_aligned[i] and 
                  volume_regime[i] and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals