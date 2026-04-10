#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation
# - Williams Fractals from 1d: Bearish fractal (sell signal) and bullish fractal (buy signal)
# - 1d EMA(50) trend filter: price > EMA50 for long bias, price < EMA50 for short bias
# - Volume confirmation: current 6h volume > 1.8x 24-period average to confirm breakout strength
# - Designed for 6h timeframe: targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Works in bull/bear markets: EMA50 filter ensures we trade with daily trend direction
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "6h_1d_williams_fractal_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute 1d Williams Fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractals: need 5 bars (2 left, center, 2 right)
    # Bullish fractal: low[n] < low[n-1] and low[n] < low[n-2] and low[n] < low[n+1] and low[n] < low[n+2]
    # Bearish fractal: high[n] > high[n-1] and high[n] > high[n-2] and high[n] > high[n+1] and high[n] > high[n+2]
    n_1d = len(high_1d)
    bullish_fractal = np.full(n_1d, np.nan)
    bearish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        # Bullish fractal at i
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
        # Bearish fractal at i
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
    
    # Need 2 extra bars for fractal confirmation (Williams requirement)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_24 = pd.Series(volume_6h).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume_6h > (1.8 * avg_volume_24)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below bullish fractal level (failed breakout) or daily trend turns bearish
            if (prices['close'].iloc[i] < bullish_fractal_aligned[i] or 
                prices['close'].iloc[i] < ema_50_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above bearish fractal level (failed breakout) or daily trend turns bullish
            if (prices['close'].iloc[i] > bearish_fractal_aligned[i] or 
                prices['close'].iloc[i] > ema_50_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for fractal breakout with trend and volume filters
            if vol_spike[i]:
                # Breakout long: price closes above bullish fractal AND above daily EMA50 (bullish alignment)
                if (prices['close'].iloc[i] > bullish_fractal_aligned[i] and 
                    prices['close'].iloc[i] > ema_50_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price closes below bearish fractal AND below daily EMA50 (bearish alignment)
                elif (prices['close'].iloc[i] < bearish_fractal_aligned[i] and 
                      prices['close'].iloc[i] < ema_50_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals