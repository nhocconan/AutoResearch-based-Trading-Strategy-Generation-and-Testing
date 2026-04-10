#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d EMA200 trend filter and volume confirmation
# - Williams Fractals from 1d: bearish fractal (sell signal) when price breaks below, bullish fractal (buy signal) when price breaks above
# - 1d EMA200 as trend filter: only take longs when price > EMA200 (bullish regime), shorts when price < EMA200 (bearish regime)
# - Volume confirmation: current 6h volume > 2.0x 20-period average to confirm institutional interest
# - Designed for 6h timeframe: targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: 1d EMA200 filter ensures we trade with higher timeframe trend
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
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Williams Fractals calculation (5-bar pattern)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Initialize fractal arrays
    bearish_fractal = np.full(len(high_1d), np.nan)  # High fractal (sell signal)
    bullish_fractal = np.full(len(high_1d), np.nan)  # Low fractal (buy signal)
    
    # Williams Fractal: need 2 bars on each side (5-bar pattern)
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: middle bar has highest high
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal: middle bar has lowest low
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 6s timeframe with proper delay
    # Williams fractals need 2 extra bars for confirmation (pattern completion)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below bullish fractal (failed breakout) or EMA200
            if (prices['close'].iloc[i] < bullish_fractal_aligned[i] or 
                prices['close'].iloc[i] < ema_200_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above bearish fractal (failed breakout) or EMA200
            if (prices['close'].iloc[i] > bearish_fractal_aligned[i] or 
                prices['close'].iloc[i] > ema_200_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for fractal breakout with trend and volume filters
            if vol_spike[i]:
                # Breakout long: price closes above bullish fractal AND above EMA200 (bullish regime)
                if (not np.isnan(bullish_fractal_aligned[i]) and 
                    prices['close'].iloc[i] > bullish_fractal_aligned[i] and
                    prices['close'].iloc[i] > ema_200_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Breakout short: price closes below bearish fractal AND below EMA200 (bearish regime)
                elif (not np.isnan(bearish_fractal_aligned[i]) and 
                      prices['close'].iloc[i] < bearish_fractal_aligned[i] and
                      prices['close'].iloc[i] < ema_200_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals