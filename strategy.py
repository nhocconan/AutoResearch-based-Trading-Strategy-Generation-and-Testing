#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1w trend filter and volume confirmation
# - Williams Fractals from 1d: bearish fractal (sell signal) when middle high has two lower highs on each side
# - Bullish fractal (buy signal) when middle low has two higher lows on each side
# - 1w EMA(21) trend filter: only take longs when price > EMA21, shorts when price < EMA21
# - Volume confirmation: current 6h volume > 2.0x 20-period average to avoid false breakouts
# - Designed for 6h timeframe: targets 12-37 trades/year to avoid fee drag
# - Works in bull/bear markets: weekly EMA filter ensures we trade with higher timeframe trend
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Fractals require 2-bar confirmation delay (align_htf_to_ltf with additional_delay_bars=2)

name = "6h_1w_williams_fractal_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 10 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1d Williams Fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractals: need 5 points (2 left, center, 2 right)
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: middle high has two lower highs on each side
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal: middle low has two higher lows on each side
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align HTF data to LTF with proper delay
    # Fractals need 2 extra bars for confirmation (2 right-side bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        close_price = prices['close'].values[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA (trend change)
            if close_price < ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA (trend change)
            if close_price > ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for fractal breakout with trend and volume filters
            if vol_spike[i]:
                # Breakout long: bullish fractal confirmed and price above weekly EMA
                if not np.isnan(bullish_fractal_aligned[i]) and close_price > ema_21_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: bearish fractal confirmed and price below weekly EMA
                elif not np.isnan(bearish_fractal_aligned[i]) and close_price < ema_21_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals