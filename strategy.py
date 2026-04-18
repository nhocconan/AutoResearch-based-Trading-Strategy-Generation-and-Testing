#!/usr/bin/env python3
"""
4h Volume-Weighted ATR Breakout with Volatility Regime Filter
Hypothesis: Breakouts from ATR-based channels during low volatility regimes (low ATR percentile)
capture explosive moves with higher reliability. Volume confirmation filters false breakouts.
Works in both bull and bear markets by adapting to volatility regimes and requiring volume.
Designed for 20-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR-based volatility channel (14-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Upper and lower channels: close ± 1.5 * ATR
    upper_channel = close + 1.5 * atr
    lower_channel = close - 1.5 * atr
    
    # Volatility regime filter: ATR percentile (50-period lookback)
    atr_series = pd.Series(atr)
    atr_percentile = atr_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Volume confirmation: 1.8x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 60  # Need enough data for ATR and percentile calculations
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or np.isnan(atr_percentile[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_pct = atr_percentile[i]
        
        # Only trade in low volatility regimes (ATR percentile < 40%)
        if atr_pct >= 0.4:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper channel with volume confirmation
            if price > upper_channel[i] and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel with volume confirmation
            elif price < lower_channel[i] and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to midline (close) or volatility increases
            if price < close[i] or atr_percentile[i] >= 0.6:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to midline (close) or volatility increases
            if price > close[i] or atr_percentile[i] >= 0.6:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_VolumeWeighted_ATRBreakout_VolRegime"
timeframe = "4h"
leverage = 1.0