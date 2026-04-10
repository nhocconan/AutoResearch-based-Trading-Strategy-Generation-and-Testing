#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d volume confirmation and ATR-based position sizing
# - Long when price breaks above upper BB(20,2) AND BB width < 20th percentile (squeeze) AND 1d volume > 1.5x 20-period average
# - Short when price breaks below lower BB(20,2) AND BB width < 20th percentile AND 1d volume > 1.5x 20-period average
# - Position size scaled by ATR volatility (0.20-0.30 range) to normalize risk
# - Exit when price returns to BB middle (mean reversion within the band)
# - Works in bull/bear: volatility contraction precedes expansion in both regimes, volume confirms institutional participation

name = "4h_1d_bb_squeeze_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h Bollinger Bands (20,2)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # BB middle (SMA20)
    sma_20 = np.full_like(close, np.nan, dtype=float)
    for i in range(19, len(close)):
        sma_20[i] = np.mean(close[i-19:i+1])
    
    # BB standard deviation
    bb_std = np.full_like(close, np.nan, dtype=float)
    for i in range(19, len(close)):
        bb_std[i] = np.std(close[i-19:i+1])
    
    # BB upper and lower
    bb_upper = sma_20 + 2.0 * bb_std
    bb_lower = sma_20 - 2.0 * bb_std
    bb_width = bb_upper - bb_lower  # Band width
    
    # Pre-compute BB width percentile (20-period lookback for regime)
    bb_width_pct = np.full_like(bb_width, np.nan, dtype=float)
    for i in range(39, len(bb_width)):  # Need 20 BB width values + 20 for lookback
        window = bb_width[i-19:i+1]  # Last 20 BB width values
        if not np.all(np.isnan(window)):
            current_width = bb_width[i]
            if not np.isnan(current_width):
                # Calculate percentile rank (0-100)
                sorted_window = np.sort(window[~np.isnan(window)])
                if len(sorted_window) > 0:
                    percentile = (np.searchsorted(sorted_window, current_width) / len(sorted_window)) * 100
                    bb_width_pct[i] = percentile
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Pre-compute 4h ATR(14) for position sizing
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR calculation (Wilder smoothing)
    atr = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= 14:
        # Initial ATR (simple average)
        atr[13] = np.nanmean(tr[1:14])
        # Wilder smoothing
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Align HTF indicators to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    bb_width_pct_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct)  # Using 1d index for BB width percentile
    atr_aligned = align_htf_to_ltf(prices, prices, atr)  # 4h ATR already in correct timeframe
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # Start after warmup (need BB width percentile lookback)
        # Skip if any required data is invalid
        if (np.isnan(close[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_width_pct_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_series = prices['volume'].values
        vol_ma_4h = np.full_like(vol_series, np.nan, dtype=float)
        # Only calculate if we have enough lookback
        if i >= 19:
            vol_ma_4h[i] = np.mean(vol_series[i-19:i+1])
        vol_spike = not np.isnan(vol_ma_4h[i]) and vol_series[i] > 1.5 * vol_ma_4h[i]
        
        # Squeeze condition: BB width < 20th percentile (low volatility regime)
        is_squeeze = bb_width_pct_aligned[i] < 20.0
        
        # Breakout conditions
        breakout_up = close[i] > bb_upper[i]
        breakout_down = close[i] < bb_lower[i]
        
        # Mean reversion exit: return to middle band
        return_to_middle = (abs(close[i] - sma_20[i]) < 0.5 * bb_width[i])  # Within half band width
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: bullish breakout AND squeeze AND volume spike
            if breakout_up and is_squeeze and vol_spike:
                # Scale position size by ATR volatility (normalize to 0.20-0.30 range)
                # Higher ATR = lower volatility = smaller position (inverse scaling)
                atr_norm = np.clip(atr_aligned[i] / np.nanmedian(atr_aligned[max(0, i-100):i+1]), 0.5, 2.0)
                size = 0.30 / atr_norm  # Inverse relationship: higher vol = smaller size
                size = np.clip(size, 0.20, 0.30)  # Keep in target range
                position = 1
                signals[i] = size
            # Short conditions: bearish breakout AND squeeze AND volume spike
            elif breakout_down and is_squeeze and vol_spike:
                atr_norm = np.clip(atr_aligned[i] / np.nanmedian(atr_aligned[max(0, i-100):i+1]), 0.5, 2.0)
                size = 0.30 / atr_norm
                size = np.clip(size, 0.20, 0.30)
                position = -1
                signals[i] = -size
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: return to middle band (mean reversion)
            if return_to_middle:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25  # Maintain position
                else:
                    signals[i] = -0.25  # Maintain position
    
    return signals