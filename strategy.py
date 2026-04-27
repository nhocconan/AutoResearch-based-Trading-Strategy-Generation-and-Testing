#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation.
- Williams Fractal identifies key swing points (resistance/support) on 6h chart
- Breakout above bearish fractal or below bullish fractal captures momentum
- 1d EMA34 trend filter ensures alignment with higher timeframe trend
- Volume spike (>1.5x 20-period average) confirms institutional participation
- Designed for 60-120 total trades over 4 years (15-30/year) to minimize fee drag
- Uses discrete position sizing (0.25) to reduce churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (high) and bullish (low)"""
    n = len(high)
    bearish = np.full(n, np.nan)  # Resistance fractal
    bullish = np.full(n, np.nan)   # Support fractal
    
    for i in range(2, n-2):
        # Bearish fractal: high[i] is highest of 5 bars (i-2, i-1, i, i+1, i+2)
        if (high[i] >= high[i-2] and high[i] >= high[i-1] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish[i] = high[i]
        
        # Bullish fractal: low[i] is lowest of 5 bars (i-2, i-1, i, i+1, i+2)
        if (low[i] <= low[i-2] and low[i] <= low[i-1] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on daily close
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        # Simple EMA calculation
        alpha = 2.0 / (34 + 1)
        ema_34_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    
    # Align 1d EMA to 6h timeframe (waits for daily close)
    ema_34_aligned = align_ltf_to_htf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Fractals on 6h data
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high, low)
    
    # Align fractals (no extra delay needed as fractals are based on completed bars)
    bearish_fractal_aligned = align_ltf_to_htf(prices, pd.DataFrame({'high': high, 'low': low}), bearish_fractal)
    bullish_fractal_aligned = align_ltf_to_htf(prices, pd.DataFrame({'high': high, 'low': low}), bullish_fractal)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(40, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above bearish fractal (resistance) + volume spike + price > 1d EMA34
            if (not np.isnan(bearish_fractal_aligned[i]) and 
                close[i] > bearish_fractal_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below bullish fractal (support) + volume spike + price < 1d EMA34
            elif (not np.isnan(bullish_fractal_aligned[i]) and 
                  close[i] < bullish_fractal_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below bullish fractal (support) OR price < 1d EMA34
            if (not np.isnan(bullish_fractal_aligned[i]) and 
                close[i] < bullish_fractal_aligned[i]) or \
               close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above bearish fractal (resistance) OR price > 1d EMA34
            if (not np.isnan(bearish_fractal_aligned[i]) and 
                close[i] > bearish_fractal_aligned[i]) or \
               close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0