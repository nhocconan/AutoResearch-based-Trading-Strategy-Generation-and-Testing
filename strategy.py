#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout + 1w EMA filter + volume confirmation
# - Primary signal: Price breaks above/below weekly Williams Fractal (confirmed 2-bar delay)
# - Trend filter: 1-week EMA50 - price must be above EMA for longs, below for shorts (higher timeframe alignment)
# - Volume confirmation: 6h volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Fractals identify key swing points, EMA50 filter ensures alignment with higher timeframe trend

name = "6h_1w_fractal_breakout_ema_volume_v1"
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
    
    # Pre-compute Williams Fractals on 1w timeframe (requires 2-bar confirmation delay)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Williams Fractal: bearish = high[n-2] < high[n-1] and high[n] < high[n-1] and high[n+1] < high[n-1] and high[n+2] < high[n-1]
    # bullish = low[n-2] > low[n-1] and low[n] > low[n-1] and low[n+1] > low[n-1] and low[n+2] > low[n-1]
    bearish_fractal = np.full(len(high_1w), np.nan)
    bullish_fractal = np.full(len(low_1w), np.nan)
    
    for i in range(2, len(high_1w) - 2):
        if (high_1w[i-2] < high_1w[i-1] and 
            high_1w[i] < high_1w[i-1] and 
            high_1w[i+1] < high_1w[i-1] and 
            high_1w[i+2] < high_1w[i-1]):
            bearish_fractal[i] = high_1w[i-1]  # fractal high at i-1
        
        if (low_1w[i-2] > low_1w[i-1] and 
            low_1w[i] > low_1w[i-1] and 
            low_1w[i+1] > low_1w[i-1] and 
            low_1w[i+2] > low_1w[i-1]):
            bullish_fractal[i] = low_1w[i-1]  # fractal low at i-1
    
    # Align Fractals to primary timeframe with 2-bar additional delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
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
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below bullish fractal OR price crosses below 1w EMA50
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above bearish fractal OR price crosses above 1w EMA50
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout with volume confirmation and 1w EMA50 filter
            # Long: price breaks above bearish fractal AND volume regime AND price above 1w EMA50
            if (close[i] > bearish_fractal_aligned[i] and 
                volume_regime[i] and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below bullish fractal AND volume regime AND price below 1w EMA50
            elif (close[i] < bullish_fractal_aligned[i] and 
                  volume_regime[i] and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals