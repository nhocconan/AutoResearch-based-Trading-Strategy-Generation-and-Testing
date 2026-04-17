#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal Breakout with 1d Volume Spike and Chop Regime Filter.
Long when price breaks above latest bearish Williams fractal (resistance) with volume spike > 1.5x MA20 AND chop > 61.8 (ranging market for mean reversion to upside).
Short when price breaks below latest bullish Williams fractal (support) with volume spike > 1.5x MA20 AND chop > 61.8.
Exit when price reverts to 12h EMA20 or opposite fractal break occurs.
Uses 1d for Williams fractals (more reliable on higher timeframe) and 12h for entry timing and volume/chop filters.
Target: 50-150 total trades over 4 years (12-37/year). Williams fractals provide strong support/resistance levels, volume spike confirms breakout validity, chop filter ensures we trade in ranging markets where mean reversion works best.
"""

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
    
    # Get 1d data for Williams fractals (more reliable on higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams fractals on 1d timeframe
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Align 1d fractals to 12h timeframe with 2-bar delay for confirmation (fractals need 2 bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 12h EMA20 for exit signal
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 12h volume MA20 for volume spike detection
    volume_s = pd.Series(volume)
    volume_ma20 = volume_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 12h Chopiness Index (CHOP) for regime filter
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = np.abs(high[i] - close[i-1])
            lc = np.abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # Calculate ATR with min_periods
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Calculate highest high and lowest low over period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(period, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        # Calculate CHOP: 100 * log10(sum(atr)/ (hh - ll)) / log10(period)
        chop = np.full_like(close, 50.0, dtype=float)
        for i in range(period, len(close)):
            if hh[i] > ll[i]:  # Avoid division by zero
                sum_atr = np.sum(atr[i-period+1:i+1])
                chop[i] = 100 * np.log10(sum_atr) / np.log10(period) / np.log10((hh[i] - ll[i]) / sum_atr) if (hh[i] - ll[i]) > 0 else 50.0
            else:
                chop[i] = 50.0
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or np.isnan(ema20[i]) or np.isnan(volume_ma20[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma20[i]
        chop_val = chop[i]
        resist = bearish_fractal_aligned[i]  # Latest bearish fractal (resistance)
        support = bullish_fractal_aligned[i]  # Latest bullish fractal (support)
        
        # Volume spike condition: current volume > 1.5x MA20
        volume_spike = vol > 1.5 * vol_ma
        
        # Chop regime condition: CHOP > 61.8 (ranging market)
        chop_regime = chop_val > 61.8
        
        if position == 0:
            # Long: Price breaks above resistance (bearish fractal) with volume spike in choppy market
            if price > resist and volume_spike and chop_regime:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below support (bullish fractal) with volume spike in choppy market
            elif price < support and volume_spike and chop_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price reverts to EMA20 OR breaks below support (opposite fractal)
            if price <= ema20[i] or price < support:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price reverts to EMA20 OR breaks above resistance (opposite fractal)
            if price >= ema20[i] or price > resist:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0