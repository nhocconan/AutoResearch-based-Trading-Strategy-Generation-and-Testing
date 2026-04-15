#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout + 1d ADX Trend Filter + Volume Confirmation
# Uses 6h Williams fractal breakouts (bullish/bearish) as entry signals, filtered by 1d ADX > 25 (trending market)
# and volume > 1.5x average to avoid false breakouts. Works in bull/bear by only taking breakouts
# in the direction of the 1d ADX trend. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data for Williams fractal calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 10:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Load 1d data for ADX and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Fractals on 6h
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n+1] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n+1] < low[n+2]
    n_6h = len(high_6h)
    bearish_fractal = np.zeros(n_6h, dtype=bool)
    bullish_fractal = np.zeros(n_6h, dtype=bool)
    
    for i in range(2, n_6h - 2):
        if (high_6h[i-2] < high_6h[i-1] and 
            high_6h[i-1] > high_6h[i] and 
            high_6h[i] > high_6h[i+1] and 
            high_6h[i+1] > high_6h[i+2]):
            bearish_fractal[i] = True
        if (low_6h[i-2] > low_6h[i-1] and 
            low_6h[i-1] < low_6h[i] and 
            low_6h[i] < low_6h[i+1] and 
            low_6h[i+1] < low_6h[i+2]):
            bullish_fractal[i] = True
    
    # Convert to arrays: 1 at fractal point, 0 otherwise
    bearish_fractal_val = bearish_fractal.astype(float)
    bullish_fractal_val = bullish_fractal.astype(float)
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    plus_dm[0] = 0
    minus_dm[0] = 0
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    # Williams fractals need 2 extra bars for confirmation (wait for 2 subsequent 6b candles)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_6h, bullish_fractal_val, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_6h, bearish_fractal_val, additional_delay_bars=2)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: bullish fractal breakout + ADX > 25 (trending) + volume spike
        if (bullish_fractal_aligned[i] > 0 and
            adx_aligned[i] > 25 and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bearish fractal breakout + ADX > 25 (trending) + volume spike
        elif (bearish_fractal_aligned[i] > 0 and
              adx_aligned[i] > 25 and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite fractal signal or ADX < 20 (ranging market)
        elif position == 1 and (bearish_fractal_aligned[i] > 0 or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bullish_fractal_aligned[i] > 0 or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsFractal_ADX_Volume"
timeframe = "6h"
leverage = 1.0