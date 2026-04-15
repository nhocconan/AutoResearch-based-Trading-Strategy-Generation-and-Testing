#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily HTF data once before loop (6h primary, 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate Williams Fractals on daily timeframe
    # Bearish fractal: high[i] is highest among [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest among [i-2, i-1, i, i+1, i+2]
    bearish_fractal = np.full(len(daily_high), np.nan, dtype=float)
    bullish_fractal = np.full(len(daily_low), np.nan, dtype=float)
    
    for i in range(2, len(daily_high) - 2):
        if (daily_high[i] >= daily_high[i-1] and daily_high[i] >= daily_high[i-2] and
            daily_high[i] >= daily_high[i+1] and daily_high[i] >= daily_high[i+2]):
            bearish_fractal[i] = daily_high[i]
        if (daily_low[i] <= daily_low[i-1] and daily_low[i] <= daily_low[i-2] and
            daily_low[i] <= daily_low[i+1] and daily_low[i] <= daily_low[i+2]):
            bullish_fractal[i] = daily_low[i]
    
    # Williams Fractals need 2 extra daily bars for confirmation
    bearish_fractal_6h = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_6h = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.concatenate([[close[0]], close[:-1]])))
    tr3 = pd.Series(np.abs(low - np.concatenate([[close[0]], close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_6h[i]) or np.isnan(bullish_fractal_6h[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Price breaks above/below confirmed Williams Fractal level
        # 2. Volume confirmation: current volume > 1.5x 20-period average
        # 3. ATR filter: ATR > 0.3% of price (avoid extremely low volatility)
        # 4. Discrete position sizing: 0.25
        
        # Volume ratio (current vs 20-period average)
        vol_ma_20 = np.nan
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        volume_ratio = volume[i] / (vol_ma_20 + 1e-10) if not np.isnan(vol_ma_20) else 0
        
        # Long conditions: break above bearish fractal (resistance) with volume
        if (close[i] > bearish_fractal_6h[i] and          # Break above resistance fractal
            volume_ratio > 1.5 and                       # Volume confirmation
            atr_14[i] > 0.003 * close[i]):               # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: break below bullish fractal (support) with volume
        elif (close[i] < bullish_fractal_6h[i] and       # Break below support fractal
              volume_ratio > 1.5 and                     # Volume confirmation
              atr_14[i] > 0.003 * close[i]):             # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Williams_Fractal_Breakout_Volume_ATR_Filter"
timeframe = "6h"
leverage = 1.0