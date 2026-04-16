#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Williams Fractal breakouts with 1w EMA50 trend filter and volume spike confirmation.
# Long when price breaks above latest bearish fractal (resistance) AND close > 1w EMA50 (uptrend) AND volume > 1.5 * avg volume.
# Short when price breaks below latest bullish fractal (support) AND close < 1w EMA50 (downtrend) AND volume > 1.5 * avg volume.
# Exit when price crosses 1w EMA50 in opposite direction.
# Uses discrete position size 0.25. Williams Fractals identify key support/resistance levels.
# 1w EMA50 ensures trading only with higher timeframe trend to avoid whipsaws in ranging markets.
# Volume spike confirms breakout validity. 12h timeframe targets 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Williams Fractals
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Get 1w data once before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 12h Indicators: Williams Fractals ===
    # Bearish fractal: high[i] is highest among [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest among [i-2, i-1, i, i+1, i+2]
    n_12h = len(high_12h)
    bearish_fractal_12h = np.full(n_12h, np.nan)
    bullish_fractal_12h = np.full(n_12h, np.nan)
    
    for i in range(2, n_12h - 2):
        if (high_12h[i] >= high_12h[i-2] and high_12h[i] >= high_12h[i-1] and
            high_12h[i] >= high_12h[i+1] and high_12h[i] >= high_12h[i+2]):
            bearish_fractal_12h[i] = high_12h[i]
        if (low_12h[i] <= low_12h[i-2] and low_12h[i] <= low_12h[i-1] and
            low_12h[i] <= low_12h[i+1] and low_12h[i] <= low_12h[i+2]):
            bullish_fractal_12h[i] = low_12h[i]
    
    # Forward fill fractal levels to maintain until broken
    bearish_fractal_12h = pd.Series(bearish_fractal_12h).ffill().values
    bullish_fractal_12h = pd.Series(bullish_fractal_12h).ffill().values
    
    # === 1w Indicators: EMA50 for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume indicators ===
    avg_volume_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (12h)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal_12h, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal_12h, additional_delay_bars=2)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    avg_volume_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # EMA50 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(avg_volume_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        bearish_fractal = bearish_fractal_aligned[i]
        bullish_fractal = bullish_fractal_aligned[i]
        ema50 = ema50_aligned[i]
        avg_volume = avg_volume_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume spike confirmation
        volume_spike = vol > 1.5 * avg_volume if avg_volume > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < EMA50 (trend break)
            if price < ema50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > EMA50 (trend break)
            if price > ema50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above bearish fractal (resistance) AND uptrend AND volume spike
            if (price > bearish_fractal) and (price > ema50) and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below bullish fractal (support) AND downtrend AND volume spike
            elif (price < bullish_fractal) and (price < ema50) and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_12hWilliamsFractal_Breakout_1wEMA50_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0