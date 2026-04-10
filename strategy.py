#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation
# - 6h Williams Fractals: bearish fractal (high with 2 lower highs on both sides) = potential resistance
#   bullish fractal (low with 2 higher lows on both sides) = potential support
# - Breakout long: price closes above recent bearish fractal level with volume spike
#   Breakout short: price closes below recent bullish fractal level with volume spike
# - 1d ADX(14) > 25 to ensure trending market and avoid chop
# - Volume confirmation: current 6h volume > 2.0x 20-period average
# - Designed for 6h timeframe: targets 12-37 trades/year to avoid fee drag
# - Works in bull/bear markets: ADX filter ensures we trade with higher timeframe trend
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 6h Williams Fractals
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Williams Fractals: need 5 points (2 left, center, 2 right)
    bearish_fractal = np.full(n, np.nan)  # resistance level
    bullish_fractal = np.full(n, np.nan)  # support level
    
    for i in range(2, n-2):
        # Bearish fractal: high[i] is highest among high[i-2:i+3]
        if (high_6h[i] > high_6h[i-1] and high_6h[i] > high_6h[i-2] and
            high_6h[i] > high_6h[i+1] and high_6h[i] > high_6h[i+2]):
            bearish_fractal[i] = high_6h[i]
        
        # Bullish fractal: low[i] is lowest among low[i-2:i+3]
        if (low_6h[i] < low_6h[i-1] and low_6h[i] < low_6h[i-2] and
            low_6h[i] < low_6h[i+1] and low_6h[i] < low_6h[i+2]):
            bullish_fractal[i] = low_6h[i]
    
    # Forward fill fractal levels to use them as support/resistance until broken
    bearish_fractal_ff = pd.Series(bearish_fractal).ffill().values
    bullish_fractal_ff = pd.Series(bullish_fractal).ffill().values
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(bearish_fractal_ff[i]) or 
            np.isnan(bullish_fractal_ff[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price re-enters below the bearish fractal level (failed breakout)
            if close_6h[i] < bearish_fractal_ff[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters above the bullish fractal level (failed breakout)
            if close_6h[i] > bullish_fractal_ff[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for fractal breakout with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 25:
                # Breakout long: price closes above recent bearish fractal (resistance)
                if close_6h[i] > bearish_fractal_ff[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price closes below recent bullish fractal (support)
                elif close_6h[i] < bullish_fractal_ff[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals