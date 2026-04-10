#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d trend filter and volume confirmation
# - Williams Fractal (1d): bearish fractal = potential short setup, bullish fractal = potential long setup
# - 1d ADX(14) > 25 to ensure trending market and avoid choppy conditions
# - Volume confirmation: current 12h volume > 2.0x 20-period average (higher threshold for fewer trades)
# - Entry: price breaks above/below recent swing high/low from fractal with trend and volume
# - Exit: ATR-based trailing stop (3.0x ATR) or price re-enters the swing range
# - Designed for 12h timeframe: targets 12-37 trades/year to avoid fee drag
# - Williams Fractals work well in both trending and ranging markets when combined with trend filter
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "12h_1d_williams_fractal_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Williams Fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractal: 5-bar pattern
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and 
            high_1d[i-3] < high_1d[i-1] and 
            high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i-1] = True
            
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and 
            low_1d[i-3] > low_1d[i-1] and 
            low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i-1] = True
    
    # Pre-compute 1d ADX(14) for trend filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 12h data
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Align HTF indicators to LTF
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractar.astype(float))
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractar.astype(float))
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 12h volume confirmation
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (2.0 * avg_volume_20)
    
    # Pre-compute 12h ATR(14) for trailing stop
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_14 = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    swing_high = 0.0  # for tracking fractal-based swing high
    swing_low = 0.0   # for tracking fractal-based swing low
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trailing stop hit OR price re-enters below swing low (failed breakout)
            if close_12h[i] < swing_low - 3.0 * atr_14[i] or close_12h[i] < swing_low:
                position = 0
                swing_high = 0.0
                swing_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trailing stop hit OR price re-enters above swing high (failed breakout)
            if close_12h[i] > swing_high + 3.0 * atr_14[i] or close_12h[i] > swing_high:
                position = 0
                swing_high = 0.0
                swing_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for fractal breakout with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 25:
                # Breakout long: bullish fractal confirmed and price breaks above recent swing high
                if bullish_fractal_aligned[i] and close_12h[i] > swing_high:
                    # Find the most recent swing high from prior bearish fractal
                    lookback = min(50, i)  # look back up to 50 bars
                    for j in range(i-1, max(i-lookback, 0), -1):
                        if bearish_fractal_aligned[j]:
                            swing_high = high_12h[j]
                            break
                    else:
                        # If no bearish fractal found, use recent high
                        swing_high = np.max(high_12h[max(i-20, 0):i])
                    
                    if close_12h[i] > swing_high:
                        position = 1
                        entry_price = close_12h[i]
                        swing_low = np.min(low_12h[max(i-20, 0):i])  # set swing low for stop
                        signals[i] = 0.25
                
                # Breakout short: bearish fractal confirmed and price breaks below recent swing low
                elif bearish_fractal_aligned[i] and close_12h[i] < swing_low:
                    # Find the most recent swing low from prior bullish fractal
                    lookback = min(50, i)  # look back up to 50 bars
                    for j in range(i-1, max(i-lookback, 0), -1):
                        if bullish_fractal_aligned[j]:
                            swing_low = low_12h[j]
                            break
                    else:
                        # If no bullish fractal found, use recent low
                        swing_low = np.min(low_12h[max(i-20, 0):i])
                    
                    if close_12h[i] < swing_low:
                        position = -1
                        entry_price = close_12h[i]
                        swing_high = np.max(high_12h[max(i-20, 0):i])  # set swing high for stop
                        signals[i] = -0.25
    
    return signals