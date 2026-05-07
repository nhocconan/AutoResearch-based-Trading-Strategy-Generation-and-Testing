#!/usr/bin/env python3
name = "6h_1d_WilliamsFractal_Pullback_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Fractals on daily
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Need 2 extra daily bars for confirmation (fractal forms after 2 bars after center)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Daily trend filter: EMA(50) on daily close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish fractal (support) with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if bullish_fractal_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal (resistance) with volume and daily downtrend
            elif bearish_fractal_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish fractal appears (resistance) or volume drops
            if bearish_fractal_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish fractal appears (support) or volume drops
            if bullish_fractal_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Williams Fractal pullback with daily trend and volume confirmation
# - Williams Fractals on daily chart identify key support/resistance levels
# - Bullish fractal = potential support (look for longs on pullback to support)
# - Bearish fractal = potential resistance (look for shorts on pullback to resistance)
# - Enter on 6h bar when fractal confirms with volume spike and daily trend alignment
# - Exit when opposing fractal appears or volume weakens
# - Works in both bull (buy pullbacks to bullish fractals in uptrend) and bear (sell pullbacks to bearish fractals in downtrend)
# - Volume spike (2.0x average) filters for institutional participation
# - Position size 0.25 targets ~20-40 trades/year, avoiding fee drag
# - Uses Williams Fractals which are proven effective in trending markets
# - Additional 2-bar delay ensures fractal is confirmed before use
# - Designed to avoid choppy markets by requiring trend and volume confirmation